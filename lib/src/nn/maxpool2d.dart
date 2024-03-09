import 'package:gpuc_dart/src/nn/nn.dart';

class MaxPool2D implements Layer2D {
  final Dim2 kernelSize;

  final Dim2 stride;

  final Dim2 padding;

  final double padValue;

  final PadMode padMode;

  final Dim2 dilation;

  // TODO return indices

  MaxPool2D(this.kernelSize,
      {this.stride = const Dim2(rows: 1, cols: 1),
      this.padding = const Dim2(rows: 0, cols: 0),
      this.padValue = 0,
      this.padMode = PadMode.constant,
      this.dilation = const Dim2(rows: 1, cols: 1)}) {
    // TODO validate
  }

  @override
  Tensor forward(Tensor inp) {
    // TODO validate
    /*
    final ctx = Context();
    // TODO if multiple devices are available try to parallelize across devices
    try {
      final stream = CudaStream(0, context: ctx);
      final inpL = CudaList.copy(inp.data, stream: stream, context: ctx);
      final out =
          CudaList.allocate(stream, outSize(inp.size).nel, context: ctx);
      CudaFFIFunctions.maxpool2D(
          stream, out.ptr, inp.ptr, kernelSize, outS, inpS,
          stride: stride,
          dilation: dilation,
          padding: padding,
          padMode: padMode,
          padValue: padValue);
      final outTensor = Tensor.sized(outSize(inp.size));
      ctx.releaseOnErr(outTensor);
      out.copyTo(outTensor.data, stream: stream);
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
     */
    throw UnimplementedError();
  }

  Dim2 outSize2D(Dim inSize) {
    // TODO is this the right calculation?
    return inSize.twoD -
        (dilation * (kernelSize - 1)) +
        (padding * 2) ~/ stride +
        Dim2(rows: 1, cols: 1);
  }

  Dim outSize(Dim inSize) {
    return Dim([inSize.batch, inSize.channels] + outSize2D(inSize).toList());
  }
}
