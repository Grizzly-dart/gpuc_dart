import 'package:gpuc_dart/src/nn2d/nn2d.dart';

class Conv2D implements Layer2D {
  final Tensor kernel;

  final Tensor? bias;

  final Dim2 kernelSize;

  final int groups;

  final Dim2 stride;

  final Dim2 padding;

  final double pad;

  final PadMode padMode;

  final Dim2 dilation;

  Conv2D.own(this.kernel,
      {
      this.bias,
      this.groups = 1,
      this.stride = const Dim2(1, 1),
      this.dilation = const Dim2(1, 1),
      this.padding = const Dim2(0, 0),
      this.pad = 0,
      this.padMode = PadMode.constant}): kernelSize = kernel.size.to2D() {
    if(kernel.size.dims != 4) {
      throw ArgumentError('kernel must be 4D');
    }
    if(bias != null) {
      // TODO
    }
    // TODO validate weight shape
    // TODO validate bias shape
  }

  factory Conv2D(
    int inChannels,
    int outChannels,
    Dim2 kernelSize, {
    Dim2 stride = const Dim2(1, 1),
    Dim2 dilation = const Dim2(1, 1),
    double pad = 0,
    PadMode padMode = PadMode.constant,
    Dim2 padding = const Dim2(0, 0),
  }) {
    final kernel = Tensor.sized(
        Dim([outChannels, inChannels, kernelSize.rows, kernelSize.cols]));
    final bias = Tensor.sized(Dim([outChannels]));
    return Conv2D.own(
      kernel,
      bias: bias,
      stride: stride,
      dilation: dilation,
      pad: pad,
      padMode: padMode,
      padding: padding,
    );
  }

  @override
  Tensor forward(Tensor input) {
    if (input.size.channels != inChannels) {
      throw ArgumentError('input channels must be $inChannels');
    }

    int batches = input.size.batch;

    // TODO device selection
    final ctx = Context();
    // TODO if multiple devices are available try to parallelize across devices
    // TODO split batches if cannot fit in memory
    try {
      final stream = CudaStream(0, context: ctx);
      final out2DS = outSize2D(input.size);
      final outS = Dim([batches, outChannels] + out2DS.toList());
      final out = CudaList.sized(stream, outS.nel, context: ctx);
      final inpL = CudaList.copy(input.as1d, stream: stream, context: ctx);
      final kernL = CudaList.copy(kernel.as1d, stream: stream, context: ctx);
      cuda.conv2D(
          stream,
          out.ptr,
          inpL.ptr,
          kernL.ptr,
          batches,
          Dim3(outChannels, out2DS.rows, out2DS.cols),
          Dim3(inChannels, input.size.rows, input.size.cols),
          kernelSize,
          groups,
          padding,
          padMode,
          pad,
          stride,
          dilation);
      final outTensor = Tensor.sized(outS);
      ctx.releaseOnErr(outTensor);
      out.copyTo(outTensor.as1d, stream: stream);
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  int get outChannels => kernel.size[0];

  int get inChannels => kernel.size[1];

  // TODO is this correct calculation?
  Dim2 outSize2D(Dim inSize) =>
      (inSize.to2D() + (padding * 2) - (dilation * (kernelSize - 1))) ~/ stride;
}
