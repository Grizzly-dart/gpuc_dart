import 'dart:async';
import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/native/cuda/cuda_extension_split.dart';

mixin Tensor2dMixin implements Tensor {
  @override
  Future<Tensor> sumRows({int colDims = 1}) async {
    if (size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    Dim inpSize = size.squeeze2D(colDims: colDims);
    Dim outSize = Dim2(inpSize.rows, 1);
    final ctx = Context();
    try {
      // TODO implement Dart summing for web
      // TODO implement C summing for non-web
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      final inp = CudaList.copy(as1d, stream: stream, context: ctx);
      final out = CudaList.sized(stream, outSize.nel, context: ctx);
      cuda.sum2D(stream, out.ptr.cast(), inp.ptr.cast(), inpSize.to2D());
      final outTensor = Tensor.sized(outSize, name: 'sum2D($name)');
      ctx.releaseOnErr(outTensor);
      out.copyTo(outTensor.as1d, stream: stream);
      // await stream.sync();
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  @override
  Future<Tensor> t({Tensor? out}) async {
    if (size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    final outSize2D = size.to2D().t;
    final outSize =
        outSize2D.extend2D(size.asList.take(size.asList.length - 2));
    final ctx = Context();
    // TODO perform transpose on CPU if nel if low
    try {
      int deviceId = 0; // TODO implement device selection
      if (out == null) {
        out = Tensor.sized(outSize, name: 'transpose2D($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != size.nel) {
          throw ArgumentError('Size mismatch');
        }
      }
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inpSize2D = size.to2D();
      final stream = CudaStream(deviceId, context: ctx);
      final inp = CudaList.copy(as1d, context: ctx);
      final outData = CudaList.sized(stream, outSize.nel, context: ctx);
      cuda.transpose2D(stream, outData.ptr.cast(), inp.ptr.cast(),
          Dim3(size.numMatrices, size.rows, size.cols));
      outData.copyTo(out.as1d, stream: stream);
      await stream.sync();
      out.size = outSize;
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  @override
  Future<Tensor> matmul(FutureOr<Tensor> other, {Tensor? out}) async {
    if (cuda.exists()) {
      int deviceId = 0; // TODO implement device selection
      return await cuda.splitMatmul(deviceId, this, await other, out: out);
    }
    // TODO if CFFI exists, use it
    // TODO implement Dart
    throw UnimplementedError();
  }

  @override
  Future<Tensor> matmulT(FutureOr<Tensor> other, {Tensor? out}) async {
    final b = await other;
    final inp1Size2D = size.to2D();
    final inp2Size2D = b.size.to2D().t;
    if (inp1Size2D.cols != inp2Size2D.rows) {
      throw ArgumentError('Columns of A must match rows of B');
    }
    if (size.numMatrices != b.size.numMatrices) {
      throw ArgumentError('Number of matrices must match');
    }
    final outSize2D = Dim2(size.rows, inp2Size2D.cols);
    final outSize =
        outSize2D.extend2D(size.asList.take(size.asList.length - 2));
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      if (out == null) {
        out = Tensor.sized(outSize, name: '$name * ${b.name}');
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
      } else if (batchSize > size.numMatrices) {
        batchSize = size.numMatrices;
      }
      final streams = <CudaStream>[];
      int batchStart = 0;
      while (batchStart < size.numMatrices) {
        int split = min(batchSize, size.numMatrices - batchStart);
        final stream = CudaStream(deviceId, context: ctx);
        streams.add(stream);
        final inp1 = CudaList.copy(
            as1d.view(batchStart * batchSize * inp1Size2D.nel,
                split * inp1Size2D.nel),
            stream: stream,
            context: ctx);
        final inp2 = CudaList.copy(
            b.as1d.view(batchStart * batchSize * inp2Size2D.nel,
                split * inp2Size2D.nel),
            stream: stream,
            context: ctx);
        final outMat =
            CudaList.sized(stream, split * outSize2D.nel, context: ctx);
        cuda.matmulT(stream, outMat.ptr.cast(), inp1.ptr, inp2.ptr, size.rows,
            size.cols, inp2Size2D.cols, split);
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

  @override
  Future<Tensor> matmulCadd(FutureOr<Tensor> other, FutureOr<Tensor> c,
      {Tensor? out}) async {
    if (cuda.exists()) {
      int deviceId = 0; // TODO implement device selection
      return await cuda.splitMatmulCadd(deviceId, this, await other, await c,
          out: out);
    }
    // TODO if CFFI exists, use it
    // TODO implement Dart
    throw UnimplementedError();
  }

  @override
  Future<Tensor> matmulCaddT(FutureOr<Tensor> other, FutureOr<Tensor> c,
      {Tensor? out}) {
    throw UnimplementedError();
  }
}
