import 'dart:async';
import 'package:gpuc_dart/gpuc_dart.dart';

class MaxPool2D<T extends num> extends Layer2D<T> {
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
  Future<Tensor<T>> compute(FutureOr<Tensor<T>> input,
      {covariant Tensor<T>? out, bool training = false}) async {
    final inp = await input;
    final outS2d = outSize2D(inp.size);
    final outS = outSize(inp.size);
    if (out != null) {
      if (out.size != outS) {
        throw ArgumentError('output size must be $outS');
      }
    }
    // TODO validate

    // TODO device selection
    final ctx = Context();
    // TODO if multiple devices are available try to parallelize across devices
    try {
      final stream = CudaStream(0);
      final inpCuda = CuOnesor<T>.copy(stream, inp.as1d, context: ctx);
      final outCuda =
          CuOnesor<T>.sized(stream, inp.type, outS2d.nel, context: ctx);
      cuda.maxPool2D(stream, outCuda.ptr, inpCuda.ptr,
          kernSize: kernelSize,
          outSize: outS2d,
          inpSize: Dim2(inp.size.rows, inp.size.cols),
          stride: stride,
          dilation: dilation,
          padding: padding,
          matrices: inp.size.asList.take(inp.size.dims - 2).prod);
      if (out == null) {
        out = Tensor.sized(outS, inp.type);
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

  Dim2 outSize2D(Dim inSize) =>
      (inSize.to2D() + (padding * 2) - (dilation * (kernelSize - 1))) ~/
          stride +
      Dim2(1, 1);

  Dim outSize(Dim inSize) {
    return Dim([inSize.batch, inSize.channels] + outSize2D(inSize).toList());
  }

  @override
  Future<Tensor> computeBackward(
      Tensor input, Tensor djByDy, Optimizer optimizer) {
    // TODO
    throw UnimplementedError();
  }
}
