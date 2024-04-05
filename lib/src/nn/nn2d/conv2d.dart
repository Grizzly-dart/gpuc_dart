import 'dart:async';
import 'package:gpuc_dart/gpuc_dart.dart';

class Conv2D extends Layer2D<double> {
  final F64Tensor kernel;

  final F64Tensor? bias;

  final Dim2 kernelSize;

  final int groups;

  final Dim2 stride;

  final Dim2 padding;

  final double pad;

  final PadMode padMode;

  final Dim2 dilation;

  final bool padSameSize;

  Conv2D.withWeights(this.kernel,
      {this.bias,
      this.groups = 1,
      this.stride = const Dim2(1, 1),
      this.dilation = const Dim2(1, 1),
      this.padding = const Dim2(0, 0),
      this.pad = 0,
      this.padMode = PadMode.constant,
      this.padSameSize = false})
      : kernelSize = kernel.size.to2D() {
    if (kernel.size.dims < 2) {
      throw ArgumentError('kernel must be at least 2D');
    } else if (kernel.size.dims > 4) {
      throw ArgumentError('kernel must be at most 4D');
    } else if (kernel.size.dims < 4) {
      kernel.size = kernel.size.ensureDims(4);
    }

    if (bias != null) {
      // TODO validate bias shape
    }
    if (outChannels % groups != 0) {
      throw ArgumentError('outChannels must be divisible by groups');
    }
    if (inChannels % groups != 0) {
      throw ArgumentError('inChannels must be divisible by groups');
    }
  }

  factory Conv2D(
    int inChannels,
    int outChannels,
    Dim2 kernelSize, {
    Dim2 stride = const Dim2(1, 1),
    Dim2 dilation = const Dim2(1, 1),
    double pad = 0,
    bool sameSize = false,
    PadMode padMode = PadMode.constant,
    Dim2 padding = const Dim2(0, 0),
  }) {
    // TODO generate random values for kernel and bias
    final kernel = F64Tensor.sized(
        Dim([outChannels, inChannels, kernelSize.rows, kernelSize.cols]));
    final bias = F64Tensor.sized(Dim([outChannels]));
    return Conv2D.withWeights(
      kernel,
      bias: bias,
      stride: stride,
      dilation: dilation,
      padSameSize: sameSize,
      pad: pad,
      padMode: padMode,
      padding: padding,
    );
  }

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor<double>> input,
      {covariant Tensor<double>? out, bool training = false}) async {
    final inp = await input;
    if (inp.size.channels != inChannels) {
      throw ArgumentError('input channels must be $inChannels');
    }
    final out2DS = outSize2D(inp.size);
    int batches = inp.size.batch;
    final outS = Dim([batches, outChannels] + out2DS.toList());
    if (out != null) {
      if (out.size != outS) {
        throw ArgumentError('output size must be $outS');
      }
    }
    Dim2 padding = this.padding;
    if (padSameSize) {
      padding = padSameSize2D(inp.size.to2D());
    }

    // TODO device selection
    final ctx = Context();
    // TODO if multiple devices are available try to parallelize across devices
    // TODO split batches if cannot fit in memory
    try {
      final stream = CudaStream(0);
      final outBuf = F64CuOnesor.sized(stream, outS.nel, context: ctx);
      final inpBuf = F64CuOnesor.copy(stream, inp.as1d, context: ctx);
      final kernBuf = F64CuOnesor.copy(stream, kernel.as1d, context: ctx);
      cuda.conv2D(
          stream,
          outBuf.ptr,
          inpBuf.ptr,
          kernBuf.ptr,
          batches,
          Dim3(outChannels, out2DS.rows, out2DS.cols),
          Dim3(inChannels, inp.size.rows, inp.size.cols),
          kernelSize,
          groups,
          padding,
          padMode,
          pad,
          stride,
          dilation);
      if(out == null) {
        out = F64Tensor.sized(outS);
        ctx.releaseOnErr(out);
      }
      outBuf.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  int get outChannels => kernel.size[0];

  int get inChannels => kernel.size[1] * groups;

  // TODO is this correct calculation?
  Dim2 outSize2D(Dim inSize) {
    Dim2 padding = this.padding;
    if (padSameSize) {
      padding = padSameSize2D(inSize.to2D());
    }
    return (inSize.to2D() +
                (padding * 2) -
                (dilation * (kernelSize - 1)) -
                Dim2(1, 1)) ~/
            stride +
        Dim2(1, 1);
  }

  Dim2 padSameSize2D(Dim2 inSize) =>
      (inSize * stride - stride + dilation * (kernelSize - 1) - inSize + 1) ~/
      2;


  @override
  Future<Tensor> computeBackward(
      Tensor input, Tensor djByDy, Optimizer optimizer) {
    // TODO
    throw UnimplementedError();
  }
}
