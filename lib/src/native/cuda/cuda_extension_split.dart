import 'dart:async';
import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/onesor/onesor.dart';

typedef CuOp1d2i3t = void Function(
    CudaStream stream, NumPtr out, NumPtr inp1, NumPtr inp2, int size);
typedef CuOp1d1i1t = void Function(
    CudaStream stream, NumPtr out, NumPtr inp1, int size);

extension CudaArithSplitExtension on Cuda {
  Future<Tensor> op1d2i3t(int deviceId, Tensor a, Tensor b, CuOp1d2i3t op,
      {Tensor? out}) async {
    if (a.nel != b.nel) throw ArgumentError('Size mismatch');
    if (out != null && out.nel != a.nel) {
      throw ArgumentError('Size mismatch');
    }
    NumType outType =
        out?.type ?? (a.type.bytes > b.type.bytes ? a.type : b.type);
    final size = a.size;
    final ctx = Context();
    try {
      if (out == null) {
        out = Tensor.sized(size, outType, name: '${a.name} + ${b.name}');
        ctx.releaseOnErr(out);
      }
      final props = cuda.getMemInfo(deviceId);
      int batchSize =
          props.total ~/ (a.lengthBytes + b.lengthBytes + out.lengthBytes);
      if (batchSize < 1) {
        throw StateError('Insufficient memory');
      } else if (batchSize > size.nel) {
        batchSize = size.nel;
      }
      final streams = <CudaStream>[];
      int batchStart = 0;
      while (batchStart < size.nel) {
        int split = min(batchSize, size.nel - batchStart);
        final stream = CudaStream(deviceId, context: ctx);
        streams.add(stream);
        final aSplit =
            CuOnesor.copy(stream, a.as1d.view(batchStart, split), context: ctx);
        final bSplit =
            CuOnesor.copy(stream, b.as1d.view(batchStart, split), context: ctx);
        final outSplit = CuOnesor.sized(stream, outType, split);
        op(stream, outSplit, aSplit, bSplit, split);
        outSplit.copyTo(out.as1d.view(batchStart, split), stream: stream);
        aSplit.release(stream: stream);
        bSplit.release(stream: stream);
        outSplit.release(stream: stream);
        batchStart += split;
      }
      await Future.wait(streams.map((s) => s.sync()));
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor> divSplit(int deviceId, Tensor a, Tensor b,
      {Tensor? out}) async {
    if (out != null) {
      if (!out.type.isFloat) {
        throw ArgumentError('Output type must be float');
      }
    } else {
      // TODO release this
      out = Tensor.sized(a.size, f64, name: '${a.name} / ${b.name}');
    }
    return op1d2i3t(deviceId, a, b, cuda.div, out: out);
  }
}

extension CudaUnarySplitExtension on Cuda {
  Future<Tensor> op1d1i1t<T>(int deviceId, Tensor a, CuOp1d1i1t op,
      {Tensor? out}) async {
    if (out != null && out.nel != a.nel) {
      throw ArgumentError('Size mismatch');
    }
    NumType outType = out?.type ?? a.type;
    final size = a.size;
    final ctx = Context();
    try {
      if (out == null) {
        out = Tensor.sized(size, outType, name: 'op(${a.name})');
        ctx.releaseOnErr(out);
      }
      final props = cuda.getMemInfo(deviceId);
      int batchSize = props.total ~/ (a.lengthBytes + out.lengthBytes);
      if (batchSize < 1) {
        throw StateError('Insufficient memory');
      } else if (batchSize > size.nel) {
        batchSize = size.nel;
      }
      final streams = <CudaStream>[];
      int batchStart = 0;
      while (batchStart < size.nel) {
        int split = min(batchSize, size.nel - batchStart);
        final stream = CudaStream(deviceId, context: ctx);
        streams.add(stream);
        final aSplit =
            CuOnesor.copy(stream, a.as1d.view(batchStart, split), context: ctx);
        final outSplit = CuOnesor.sized(stream, outType, split);
        op(stream, outSplit, aSplit, split);
        outSplit.copyTo(out.as1d.view(batchStart, split), stream: stream);
        aSplit.release(stream: stream);
        outSplit.release(stream: stream);
        batchStart += split;
      }
      await Future.wait(streams.map((s) => s.sync()));
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }
}

extension CudaSplitExtension on Cuda {
  // TODO use tensor views instead
  Future<Tensor<double>> matmulSplit(
      int deviceId, Tensor<double> a, Tensor<double> b,
      {Tensor<double>? out}) async {
    if (a.size.cols != b.size.rows) {
      throw ArgumentError('Columns of A must match rows of B');
    }
    final outSize2D = Dim2(a.size.rows, b.size.cols);
    final outSize = a.size.withMatrix(outSize2D.rows, outSize2D.cols);
    if (out != null) {
      if (out.size != outSize) {
        throw ArgumentError('Size mismatch');
      }
    }
    int numMats = a.size.numMatrices;
    if (numMats != b.size.numMatrices) {
      throw ArgumentError('Number of matrices must match');
    }

    final inp1Size2D = a.size.to2D();
    final inp2Size2D = b.size.to2D();

    final ctx = Context();
    try {
      if (out == null) {
        out = F64Tensor.sized(outSize, name: '${a.name} * ${b.name}');
        ctx.releaseOnErr(out);
      }
      final props = cuda.getMemInfo(deviceId);
      int batchSize = props.total ~/
          ((inp1Size2D.nel + inp2Size2D.nel + outSize2D.nel) * 8);
      if (batchSize < 1) {
        throw StateError('Insufficient memory');
      } else if (batchSize > numMats) {
        batchSize = numMats;
      }
      final streams = <CudaStream>[];
      int batchStart = 0;
      while (batchStart < numMats) {
        int split = min(batchSize, numMats - batchStart);
        final stream = CudaStream(deviceId, context: ctx);
        streams.add(stream);
        final inp1 = F64CuOnesor.copy(
            stream,
            a.as1d.view(batchStart * batchSize * inp1Size2D.nel,
                split * inp1Size2D.nel),
            context: ctx);
        final inp2 = F64CuOnesor.copy(
            stream,
            b.as1d.view(batchStart * batchSize * inp2Size2D.nel,
                split * inp2Size2D.nel),
            context: ctx);
        final outMat =
            F64CuOnesor.sized(stream, split * outSize2D.nel, context: ctx);
        cuda.matmul(stream, outMat.ptr.cast(), inp1.ptr, inp2.ptr, a.size.rows,
            a.size.cols, b.size.cols, split);
        outMat.copyTo(
            out.as1d.view(
                batchStart * batchSize * outSize2D.nel, split * outSize2D.nel),
            stream: stream);
        inp1.release(stream: stream);
        inp2.release(stream: stream);
        outMat.release(stream: stream);
        batchStart += split;
      }
      await Future.wait(streams.map((s) => s.sync()));
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> matmulTSplit(
      int deviceId, Tensor<double> a, Tensor<double> b,
      {Tensor<double>? out}) async {
    final inp1Size2D = a.size.to2D();
    final inp2Size2D = b.size.to2D().t;
    if (inp1Size2D.cols != inp2Size2D.rows) {
      throw ArgumentError('Columns of A must match rows of B');
    }
    final numMats = a.size.numMatrices;
    if (numMats != b.size.numMatrices) {
      throw ArgumentError('Number of matrices must match');
    }

    final outSize2D = Dim2(inp1Size2D.rows, inp2Size2D.cols);
    final outSize = a.size.withMatrix(outSize2D.rows, outSize2D.cols);
    final ctx = Context();
    try {
      if (out == null) {
        out = F64Tensor.sized(outSize, name: '${a.name} * ${b.name}');
        ctx.releaseOnErr(out);
      } else {
        if (out.size != outSize) {
          throw ArgumentError('Size mismatch');
        }
      }
      final props = cuda.getMemInfo(deviceId);
      int batchSize = props.total ~/
          ((inp1Size2D.nel + inp2Size2D.nel + outSize2D.nel) * 8);
      if (batchSize < 1) {
        throw StateError('Insufficient memory');
      } else if (batchSize > numMats) {
        batchSize = numMats;
      }
      final streams = <CudaStream>[];
      int batchStart = 0;
      while (batchStart < numMats) {
        int split = min(batchSize, numMats - batchStart);
        final stream = CudaStream(deviceId, context: ctx);
        streams.add(stream);
        final inp1 = F64CuOnesor.copy(
            stream,
            a.as1d.view(batchStart * batchSize * inp1Size2D.nel,
                split * inp1Size2D.nel),
            context: ctx);
        final inp2 = F64CuOnesor.copy(
            stream,
            b.as1d.view(batchStart * batchSize * inp2Size2D.nel,
                split * inp2Size2D.nel),
            context: ctx);
        final outMat =
            F64CuOnesor.sized(stream, split * outSize2D.nel, context: ctx);
        cuda.matmulT(stream, outMat.ptr.cast(), inp1.ptr, inp2.ptr, a.size.rows,
            a.size.cols, inp2Size2D.cols, split);
        outMat.copyTo(
            out.as1d.view(
                batchStart * batchSize * outSize2D.nel, split * outSize2D.nel),
            stream: stream);
        inp1.release(stream: stream);
        inp2.release(stream: stream);
        outMat.release(stream: stream);
        batchStart += split;
      }
      await Future.wait(streams.map((s) => s.sync()));
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> splitMatmulCadd(
      int deviceId, Tensor<double> a, Tensor<double> b, Tensor<double> add,
      {Tensor<double>? out}) async {
    if (a.size.cols != b.size.rows) {
      throw ArgumentError('Columns of A must match rows of B');
    }
    if (add.nel != b.size.cols) {
      throw ArgumentError(
          'Add vector should have same number as B Tensor column');
    }
    int numMats = a.size.numMatrices;
    if (numMats != b.size.numMatrices) {
      throw ArgumentError('Number of matrices must match');
    }

    final inp1Size2D = a.size.to2D();
    final inp2Size2D = b.size.to2D();
    final outSize2D = Dim2(a.size.rows, b.size.cols);
    final outSize =
        outSize2D.extend2D(a.size.asList.take(a.size.asList.length - 2));

    final ctx = Context();
    try {
      final stream = CudaStream(deviceId, context: ctx);
      final addCuda = F64CuOnesor.copy(stream, add.as1d, context: ctx);
      if (out == null) {
        out = F64Tensor.sized(outSize, name: '${a.name} * ${b.name}');
        ctx.releaseOnErr(out);
      } else {
        if (out.size != outSize) {
          throw ArgumentError('Size mismatch');
        }
      }
      final props = cuda.getMemInfo(deviceId);
      int batchSize = props.total ~/
          ((inp1Size2D.nel + inp2Size2D.nel + outSize2D.nel) * 8);
      if (batchSize < 1) {
        throw StateError('Insufficient memory');
      } else if (batchSize > numMats) {
        batchSize = numMats;
      }
      final streams = <CudaStream>[];
      int batchStart = 0;
      while (batchStart < numMats) {
        int split = min(batchSize, numMats - batchStart);
        final stream = CudaStream(deviceId, context: ctx);
        streams.add(stream);
        final inp1 = F64CuOnesor.copy(
            stream,
            a.as1d.view(batchStart * batchSize * inp1Size2D.nel,
                split * inp1Size2D.nel),
            context: ctx);
        final inp2 = F64CuOnesor.copy(
            stream,
            b.as1d.view(batchStart * batchSize * inp2Size2D.nel,
                split * inp2Size2D.nel),
            context: ctx);
        final outMat =
            F64CuOnesor.sized(stream, split * outSize2D.nel, context: ctx);
        cuda.matmulCadd(stream, outMat.ptr.cast(), inp1.ptr, inp2.ptr,
            addCuda.ptr, a.size.rows, a.size.cols, b.size.cols, split);
        outMat.copyTo(
            out.as1d.view(
                batchStart * batchSize * outSize2D.nel, split * outSize2D.nel),
            stream: stream);
        inp1.release(stream: stream);
        inp2.release(stream: stream);
        outMat.release(stream: stream);
        batchStart += split;
      }
      await Future.wait(streams.map((s) => s.sync()));
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> splitMatmulTCadd(
      int deviceId, Tensor<double> a, Tensor<double> b, Tensor<double> add,
      {Tensor<double>? out}) async {
    final inp1Size2D = a.size.to2D();
    final inp2Size2D = b.size.to2D().t;
    if (inp1Size2D.cols != inp2Size2D.rows) {
      throw ArgumentError('Columns of A must match rows of B');
    }
    if (add.nel != inp2Size2D.cols) {
      throw ArgumentError(
          'Add vector should have same number as B Tensor column');
    }
    int numMats = a.size.numMatrices;
    if (numMats != b.size.numMatrices) {
      throw ArgumentError('Number of matrices must match');
    }

    final outSize2D = Dim2(a.size.rows, b.size.cols);
    final outSize = a.size.withMatrix(outSize2D.rows, outSize2D.cols);

    final ctx = Context();
    try {
      final stream = CudaStream(deviceId, context: ctx);
      final addCuda = F64CuOnesor.copy(stream, add.as1d, context: ctx);
      if (out == null) {
        out = F64Tensor.sized(outSize, name: '${a.name} * ${b.name}');
        ctx.releaseOnErr(out);
      } else {
        if (out.size != outSize) {
          throw ArgumentError('Size mismatch');
        }
      }
      final props = cuda.getMemInfo(deviceId);
      int batchSize = props.total ~/
          ((inp1Size2D.nel + inp2Size2D.nel + outSize2D.nel) * 8);
      if (batchSize < 1) {
        throw StateError('Insufficient memory');
      } else if (batchSize > numMats) {
        batchSize = numMats;
      }
      final streams = <CudaStream>[];
      int batchStart = 0;
      while (batchStart < numMats) {
        int split = min(batchSize, numMats - batchStart);
        final stream = CudaStream(deviceId, context: ctx);
        streams.add(stream);
        final inp1 = F64CuOnesor.copy(
            stream,
            a.as1d.view(batchStart * batchSize * inp1Size2D.nel,
                split * inp1Size2D.nel),
            context: ctx);
        final inp2 = F64CuOnesor.copy(
            stream,
            b.as1d.view(batchStart * batchSize * inp2Size2D.nel,
                split * inp2Size2D.nel),
            context: ctx);
        final outMat =
            F64CuOnesor.sized(stream, split * outSize2D.nel, context: ctx);
        cuda.matmulCadd(stream, outMat.ptr.cast(), inp1.ptr, inp2.ptr,
            addCuda.ptr, a.size.rows, a.size.cols, b.size.cols, split);
        outMat.copyTo(
            out.as1d.view(
                batchStart * batchSize * outSize2D.nel, split * outSize2D.nel),
            stream: stream);
        inp1.release(stream: stream);
        inp2.release(stream: stream);
        outMat.release(stream: stream);
        batchStart += split;
      }
      await Future.wait(streams.map((s) => s.sync()));
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }
}
