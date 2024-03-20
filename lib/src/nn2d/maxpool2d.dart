import 'dart:async';
import 'package:gpuc_dart/gpuc_dart.dart';

class MaxPool2D implements Layer2D<double> {
  final Dim2 kernelSize;

  final Dim2 stride;

  final Dim2 padding;

  final Dim2 dilation;

  // TODO return indices

  MaxPool2D(this.kernelSize,
      {Dim2? stride,
      this.padding = const Dim2(0, 0),
      this.dilation = const Dim2(1, 1)})
      : stride = stride ?? kernelSize {
    // TODO validate
  }

  @override
  Future<F64Tensor> forward(FutureOr<Tensor<double>> input) async {
    final inp = await input;
    // TODO validate

    // TODO device selection
    final ctx = Context();
    // TODO if multiple devices are available try to parallelize across devices
    try {
      final stream = CudaStream(0, context: ctx);
      final outS = outSize2D(inp.size);
      final inpL = F64CuOnesor.copy(inp.as1d, stream: stream, context: ctx);
      final out = F64CuOnesor.sized(stream, outS.nel, context: ctx);
      cuda.maxPool2D(stream, out.ptr, inpL.ptr,
          kernSize: kernelSize,
          outSize: outS,
          inpSize: Dim2(inp.size.rows, inp.size.cols),
          stride: stride,
          dilation: dilation,
          padding: padding,
          matrices: inp.size.asList.take(inp.size.dims - 2).prod);
      final outTensor = F64Tensor.sized(outS);
      ctx.releaseOnErr(outTensor);
      out.copyTo(outTensor.as1d, stream: stream);
      await stream.sync();
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
