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
      return await cuda.matmulSplit(deviceId, this, await other, out: out);
    }
    // TODO if CFFI exists, use it
    // TODO implement Dart
    throw UnimplementedError();
  }

  @override
  Future<Tensor> matmulT(FutureOr<Tensor> other, {Tensor? out}) async {
    if (cuda.exists()) {
      int deviceId = 0; // TODO implement device selection
      return await cuda.matmulSplit(deviceId, this, await other, out: out);
    }
    // TODO if CFFI exists, use it
    // TODO implement Dart
    throw UnimplementedError();
  }

  @override
  Future<Tensor> matmulCadd(FutureOr<Tensor> other, FutureOr<Tensor> c,
      {Tensor? out}) async {
    if (cuda.exists()) {
      int deviceId = 0; // TODO implement device selection
      return cuda.splitMatmulCadd(deviceId, this, await other, await c,
          out: out);
    }
    // TODO if CFFI exists, use it
    // TODO implement Dart
    throw UnimplementedError();
  }

  @override
  Future<Tensor> matmulCaddT(FutureOr<Tensor> other, FutureOr<Tensor> c,
      {Tensor? out}) async {
    if (cuda.exists()) {
      int deviceId = 0; // TODO implement device selection
      return cuda.splitMatmulTCadd(deviceId, this, await other, await c,
          out: out);
    }
    // TODO if CFFI exists, use it
    // TODO implement Dart
    throw UnimplementedError();
  }
}
