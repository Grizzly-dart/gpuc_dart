import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

class ELUActivation extends Layer {
  final double alpha;

  ELUActivation({this.alpha = 1.0});

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor> input,
      {covariant Tensor<double>? out, bool training = false}) async {
    // TODO device selection
    final ctx = Context();

    final inp = await input;

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

class RELUActivation extends Layer {
  RELUActivation();

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor> input,
      {covariant Tensor<double>? out, bool training = false}) async {
    // TODO device selection
    final ctx = Context();

    final inp = await input;

    try {
      final stream = CudaStream(0, context: ctx);
      final inpCuda = CuOnesor.copy(stream, inp.as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, inpCuda.length, context: ctx);
      cuda.minThreshold(stream, outCuda, inpCuda, 0, 0, inpCuda.length);
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

class TanhActivation extends Layer {
  TanhActivation();

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor> input,
          {covariant Tensor<double>? out, bool training = false}) async =>
      (await input).tanh(out: out);
}

class ThresholdActivation extends Layer {
  double threshold;

  double value;

  ThresholdActivation({required this.threshold, required this.value});

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor> input,
      {covariant Tensor<double>? out, bool training = false}) async {
    // TODO device selection
    final ctx = Context();

    final inp = await input;

    try {
      final stream = CudaStream(0, context: ctx);
      final inpCuda = CuOnesor.copy(stream, inp.as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, inpCuda.length, context: ctx);
      cuda.minThreshold(
          stream, outCuda, inpCuda, threshold, value, inpCuda.length);
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

class SigmoidActivation extends Layer {
  SigmoidActivation();

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor> input,
      {covariant Tensor<double>? out, bool training = false}) async {
    // TODO device selection
    final ctx = Context();

    final inp = await input;

    try {
      final stream = CudaStream(0, context: ctx);
      final inpCuda = CuOnesor.copy(stream, inp.as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, inpCuda.length, context: ctx);
      cuda.sigmoidActivation(stream, outCuda, inpCuda, inpCuda.length);
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

class SiLUActivation extends Layer {
  SiLUActivation();

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor> input,
      {covariant Tensor<double>? out, bool training = false}) async {
    final inp = await input;
    if (out != null) {
      if (out.size != inp.size) {
        throw ArgumentError('output size must be equal to input size');
      }
    }
    // TODO device selection
    final ctx = Context();

    try {
      final stream = CudaStream(0, context: ctx);
      final inpCuda = CuOnesor.copy(stream, inp.as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, inpCuda.length, context: ctx);
      cuda.siluActivation(stream, outCuda, inpCuda, inpCuda.length);
      if (out == null) {
        out = F64Tensor.sized(inp.size);
        ctx.releaseOnErr(out);
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

class SoftplusActivation extends Layer {
  int beta;
  int threshold;

  SoftplusActivation({this.beta = 1, this.threshold = 20});

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor> input,
      {covariant Tensor<double>? out, bool training = false}) async {
    // TODO device selection
    final ctx = Context();

    final inp = await input;

    try {
      final stream = CudaStream(0, context: ctx);
      final inpCuda = CuOnesor.copy(stream, inp.as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, inpCuda.length, context: ctx);
      cuda.softplusActivation(
          stream, outCuda, inpCuda, inpCuda.length, beta, threshold);
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

class SoftsignActivation extends Layer {
  SoftsignActivation();

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor> input,
      {covariant Tensor<double>? out, bool training = false}) async {
    // TODO device selection
    final ctx = Context();

    final inp = await input;

    try {
      final stream = CudaStream(0, context: ctx);
      final inpCuda = CuOnesor.copy(stream, inp.as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, inpCuda.length, context: ctx);
      cuda.softsignActivation(stream, outCuda, inpCuda, inpCuda.length);
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

class MishActivation extends Layer {
  MishActivation();

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor> input,
      {covariant Tensor<double>? out, bool training = false}) async {
    final inp = await input;
    if (out != null) {
      if (out.size != inp.size) {
        throw ArgumentError('output size must be equal to input size');
      }
    }
    // TODO device selection
    final ctx = Context();

    try {
      final stream = CudaStream(0, context: ctx);
      final inpCuda = CuOnesor.copy(stream, inp.as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, inpCuda.length, context: ctx);
      cuda.mishActivation(stream, outCuda, inpCuda, inpCuda.length);
      if (out == null) {
        out = F64Tensor.sized(inp.size);
        ctx.releaseOnErr(out);
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
