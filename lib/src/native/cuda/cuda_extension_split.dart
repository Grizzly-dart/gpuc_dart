import 'dart:async';
import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/native/cuda/native_types.dart';

// TODO pass in device selection

typedef CuOp1d2i = void Function(
    CudaStream stream, NumPtr out, NumPtr inp1, NumPtr inp2, int size);
typedef CuOp1d1i = void Function(
    CudaStream stream, NumPtr out, NumPtr inp1, int size);
typedef CuOp2d1i = void Function(
    CudaStream stream, NumPtr out, NumPtr inp1, Dim2 inpSize);

extension CudaSplitExtension on Cuda {
  Future<Tensor> op1d1i<T>(int deviceId, Tensor a, CuOp1d1i op,
      {Tensor? out, NumType? outType}) async {
    if (out != null && out.nel != a.nel) {
      throw ArgumentError('Size mismatch');
    }
    NumType oType = out?.type ?? outType ?? a.type;
    final size = a.size;
    final ctx = Context();
    try {
      out ??= Tensor.sized(size, oType, name: 'op(${a.name})');
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
        final stream = CudaStream(deviceId);
        streams.add(stream);
        final aSplit =
            CuOnesor.copy(stream, a.as1d.view(batchStart, split), context: ctx);
        final outSplit = CuOnesor.sized(stream, oType, split);
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

  Future<Tensor> op1d2i(int deviceId, Tensor a, Tensor b, OpBinaryArith op,
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
          props.total ~/ (a.type.bytes + b.type.bytes + out.type.bytes);
      if (batchSize < 1) {
        throw StateError('Insufficient memory');
      } else if (batchSize > size.nel) {
        batchSize = size.nel;
      }
      final streams = <CudaStream>[];
      int batchStart = 0;
      while (batchStart < size.nel) {
        int split = min(batchSize, size.nel - batchStart);
        final stream = CudaStream(deviceId);
        streams.add(stream);
        final aSplit =
            CuOnesor.copy(stream, a.as1d.view(batchStart, split), context: ctx);
        final bSplit =
            CuOnesor.copy(stream, b.as1d.view(batchStart, split), context: ctx);
        final outSplit = CuOnesor.sized(stream, outType, split);
        binaryArith(op, stream, outSplit, aSplit, bSplit, split);
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

  Future<Tensor> op1d2iScalar(int deviceId, Tensor a, num b, OpBinaryArith op,
      {Tensor? out, bool flip = false}) async {
    if (out != null && out.nel != a.nel) {
      throw ArgumentError('Size mismatch');
    }
    NumType outType = out?.type ?? a.type;
    final size = a.size;
    final ctx = Context();
    try {
      if (out == null) {
        out = Tensor.sized(size, outType, name: a.name);
        ctx.releaseOnErr(out);
      }
      final props = cuda.getMemInfo(deviceId);
      int batchSize = props.total ~/ (a.type.bytes + out.type.bytes);
      if (batchSize < 1) {
        throw StateError('Insufficient memory');
      } else if (batchSize > size.nel) {
        batchSize = size.nel;
      }
      final streams = <CudaStream>[];
      int batchStart = 0;
      while (batchStart < size.nel) {
        int split = min(batchSize, size.nel - batchStart);
        final stream = CudaStream(deviceId);
        streams.add(stream);
        final aSplit =
            CuOnesor.copy(stream, a.as1d.view(batchStart, split), context: ctx);
        final outSplit = CuOnesor.sized(stream, outType, split, context: ctx);
        binaryArithScalar(op, stream, outSplit, aSplit, b, split, flip: flip);
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
    return op1d2i(deviceId, a, b, cuFFI.div, out: out);
  }

  Future<Tensor> divSplitScalar(int deviceId, Tensor a, num b,
      {Tensor? out}) async {
    if (out != null) {
      if (!out.type.isFloat) {
        throw ArgumentError('Output type must be float');
      }
    } else {
      // TODO release this
      out = Tensor.sized(a.size, f64, name: '${a.name} / $b');
    }
    return op1d2iScalar(deviceId, a, b, cuFFI.div, out: out);
  }

  Future<double> op1dF64Red(int deviceId, Tensor a, CuOp1d1i op) async {
    final ctx = Context();
    try {
      final stream = CudaStream(deviceId);
      final inpCuda = CuOnesor.copy(stream, a.as1d, context: ctx);
      final out = F64CuOnesor.sized(stream, 1, context: ctx);
      op(stream, out, inpCuda, a.nel);
      final ret = out.read(context: ctx);
      await stream.sync();
      return ret[0];
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<T> op1d2tRed<T>(int deviceId, Tensor a, CuOp1d1i op) async {
    final ctx = Context();
    try {
      final stream = CudaStream(deviceId);
      final inpCuda = CuOnesor.copy(stream, a.as1d, context: ctx);
      NumType outType;
      if (a.type.isFloat) {
        outType = f64;
      } else if (a.type.isUInt) {
        outType = u64;
      } else {
        outType = i64;
      }
      final out = CuOnesor.sized(stream, outType, 1, context: ctx);
      op(stream, out, inpCuda, a.nel);
      final ret = out.read(context: ctx);
      await stream.sync();
      return ret[0] as T;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor> op2d1i<T>(int deviceId, Tensor a, CuOp2d1i op,
      {int colDims = 1, Tensor? out, NumType? outType}) async {
    if (a.size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    Dim2 inpSize = a.size.squeeze2D(colDims: colDims);
    Dim outSize = Dim2(inpSize.rows, 1);
    if (out != null && out.size.nel != outSize.nel) {
      throw ArgumentError('Size mismatch');
    }
    final oType = out?.type ?? outType ?? a.type;
    final ctx = Context();
    try {
      if (out == null) {
        out = Tensor.sized(outSize, oType, name: 'op(${a.name})');
        ctx.releaseOnErr(out);
      }
      final props = cuda.getMemInfo(deviceId);
      int batchSize =
          props.total ~/ (a.size.cols * a.bytesPerItem + out.bytesPerItem);
      if (batchSize < 1) {
        throw StateError('Insufficient memory');
      } else if (batchSize > inpSize.rows) {
        batchSize = inpSize.rows;
      }
      final streams = <CudaStream>[];
      int batchStart = 0;
      while (batchStart < inpSize.rows) {
        int split = min(batchSize, inpSize.rows - batchStart);
        final stream = CudaStream(deviceId);
        streams.add(stream);
        final aSplit = CuOnesor.copy(
            stream, a.as1d.view(batchStart * a.size.cols, split * a.size.cols),
            context: ctx);
        final outSplit = CuOnesor.sized(stream, oType, split, context: ctx);
        op(stream, outSplit, aSplit, inpSize);
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

extension CudaMatrixExtension on Cuda {
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
      out ??= F64Tensor.sized(outSize, name: '${a.name} * ${b.name}');
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
        final stream = CudaStream(deviceId);
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
    if (out != null && out.size != outSize) {
      throw ArgumentError('Size mismatch');
    }

    final ctx = Context();
    try {
      out ??= F64Tensor.sized(outSize, name: '${a.name} * ${b.name}');
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
        final stream = CudaStream(deviceId);
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
      final stream = CudaStream(deviceId);
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
        final stream = CudaStream(deviceId);
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
      final stream = CudaStream(deviceId);
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
        final stream = CudaStream(deviceId);
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
