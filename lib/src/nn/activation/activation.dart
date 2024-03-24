import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

class ELU implements Layer {
  final double alpha;

  ELU({this.alpha = 1.0});

  @override
  Future<Tensor<double>> forward(FutureOr<Tensor> x,
      {Tensor<double>? out}) async {
    // TODO device selection
    final ctx = Context();

    final inp = await x;

    try {
      final stream = CudaStream(0, context: ctx);
      final inpCuda = CuOnesor.copy(stream, inp.as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, inpCuda.length, context: ctx);
      cuda.eluActivation(stream, outCuda, inpCuda, inpCuda.length, alpha);
      if (out == null) {
        out = F64Tensor.sized(inp.size);
        ctx.releaseOnErr(out);
      } else {
        if (out.size != inp.size) {
          throw ArgumentError('output size must be equal to input size');
        }
      }
      outCuda.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }
}

class RELU implements Layer {
  RELU();

  @override
  Future<Tensor<double>> forward(FutureOr<Tensor> x,
      {Tensor<double>? out}) async {
    // TODO device selection
    final ctx = Context();

    final inp = await x;

    try {
      final stream = CudaStream(0, context: ctx);
      final inpCuda = CuOnesor.copy(stream, inp.as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, inpCuda.length, context: ctx);
      cuda.reluActivation(stream, outCuda, inpCuda, inpCuda.length);
      if (out == null) {
        out = F64Tensor.sized(inp.size);
        ctx.releaseOnErr(out);
      } else {
        if (out.size != inp.size) {
          throw ArgumentError('output size must be equal to input size');
        }
      }
      outCuda.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }
}

class Tanh implements Layer {
  Tanh();

  @override
  Future<Tensor<double>> forward(FutureOr<Tensor> x,
      {Tensor<double>? out}) async {
    return (await x).tanh(out: out);
  }
}
