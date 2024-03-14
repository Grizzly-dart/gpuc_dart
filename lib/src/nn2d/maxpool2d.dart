import 'package:gpuc_dart/src/nn2d/nn2d.dart';

class MaxPool2D implements Layer2D {
  final Dim2 kernelSize;

  final Dim2 stride;

  final Dim2 padding;

  final double pad;

  final PadMode padMode;

  final Dim2 dilation;

  // TODO return indices

  MaxPool2D(this.kernelSize,
      {Dim2? stride,
      this.padding = const Dim2(0, 0),
      this.pad = 0,
      this.padMode = PadMode.constant,
      this.dilation = const Dim2(1, 1)})
      : stride = stride ?? kernelSize {
    // TODO validate
  }

  @override
  Tensor forward(Tensor inp) {
    // TODO validate

    // TODO device selection
    final ctx = Context();
    // TODO if multiple devices are available try to parallelize across devices
    try {
      final stream = CudaStream(0, context: ctx);
      final outS = outSize2D(inp.size);
      final inpL = CudaList.copy(inp.as1d, stream: stream, context: ctx);
      final out = CudaList.sized(stream, outS.nel, context: ctx);
      cuda.maxPool2D(stream, out.ptr, inpL.ptr,
          kernSize: kernelSize,
          outSize: outS,
          inpSize: Dim2(inp.size.rows, inp.size.cols),
          stride: stride,
          dilation: dilation,
          padding: padding,
          padMode: padMode,
          pad: pad,
          matrices: inp.size.asList.skip(2).prod);
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

  Dim2 outSize2D(Dim inSize) =>
      (inSize.to2D() + (padding * 2) - (dilation * (kernelSize - 1))) ~/ stride +
      Dim2(1, 1);

  Dim outSize(Dim inSize) {
    return Dim([inSize.batch, inSize.channels] + outSize2D(inSize).toList());
  }
}
