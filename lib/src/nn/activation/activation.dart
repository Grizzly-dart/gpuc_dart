import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

class ELUActivation implements Layer {
  final double alpha;

  ELUActivation({this.alpha = 1.0});

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

class RELUActivation implements Layer {
  RELUActivation();

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

class TanhActivation implements Layer {
  TanhActivation();

  @override
  Future<Tensor<double>> forward(FutureOr<Tensor> x,
      {Tensor<double>? out}) async {
    return (await x).tanh(out: out);
  }
}

class ThresholdActivation implements Layer {
  double threshold;

  double value;

  ThresholdActivation({required this.threshold, required this.value});

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

class SigmoidActivation implements Layer {
  SigmoidActivation();

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

class SiLUActivation implements Layer {
  SiLUActivation();

  @override
  Future<Tensor<double>> forward(FutureOr<Tensor> x,
      {Tensor<double>? out}) async {
    final inp = await x;
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

class SoftplusActivation implements Layer {
  int beta;
  int threshold;

  SoftplusActivation({this.beta = 1, this.threshold = 20});

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

class SoftsignActivation implements Layer {
  SoftsignActivation();

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

class MishActivation implements Layer {
  MishActivation();

  @override
  Future<Tensor<double>> forward(FutureOr<Tensor> x,
      {Tensor<double>? out}) async {
    final inp = await x;
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
