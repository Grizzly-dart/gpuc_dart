import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/native/cuda/cuda_extension_split.dart';

mixin F64Tensor2dMixin implements F64Tensor {
  @override
  Future<Tensor<double>> t({Tensor<double>? out}) async {
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
        out = F64Tensor.sized(outSize, name: 'transpose2D($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != size.nel) {
          throw ArgumentError('Size mismatch');
        }
      }
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final stream = CudaStream(deviceId, context: ctx);
      final inp = F64CuOnesor.copy(stream, as1d, context: ctx);
      final outData = F64CuOnesor.sized(stream, outSize.nel, context: ctx);
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
  Future<Tensor<double>> matmul(FutureOr<Tensor<double>> other,
      {Tensor<double>? out}) async {
    if (cuda.exists()) {
      int deviceId = 0; // TODO implement device selection
      return await cuda.matmulSplit(deviceId, this, await other, out: out);
    }
    // TODO if CFFI exists, use it
    // TODO implement Dart
    throw UnimplementedError();
  }

  @override
  Future<Tensor<double>> matmulT(FutureOr<Tensor<double>> other,
      {Tensor<double>? out}) async {
    if (cuda.exists()) {
      int deviceId = 0; // TODO implement device selection
      return await cuda.matmulSplit(deviceId, this, await other, out: out);
    }
    // TODO if CFFI exists, use it
    // TODO implement Dart
    throw UnimplementedError();
  }

  @override
  Future<Tensor<double>> matmulCadd(
      FutureOr<Tensor<double>> other, FutureOr<Tensor<double>> c,
      {Tensor<double>? out}) async {
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
  Future<Tensor<double>> matmulCaddT(
      FutureOr<Tensor<double>> other, FutureOr<Tensor<double>> c,
      {Tensor<double>? out}) async {
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
