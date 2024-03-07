import 'package:gpuc_dart/gpuc_dart.dart';

abstract class Layer2D {
  Tensor forward(Tensor input);
}

class MaxPool2D implements Layer2D {
  final Size2D kernelSize;

  final Size2D stride;

  final Size2D padding;

  final double padValue;

  final PadMode padMode;

  final Size2D dilation;

  // TODO return indices

  MaxPool2D(this.kernelSize,
      {this.stride = const Size2D(rows: 1, cols: 1),
        this.padding = const Size2D(rows: 0, cols: 0),
        this.padValue = 0,
        this.padMode = PadMode.constant,
        this.dilation = const Size2D(rows: 1, cols: 1)}) {
    // TODO validate
  }

  @override
  Tensor forward(Tensor inp) {
    // TODO validate
    // TODO if multiple devices are available try to parallelize across devices
    /*
    if (inp.deviceType == DeviceType.cuda) {
      final out = Tensor.empty(outSize(inp.size),
          deviceType: inp.deviceType, deviceId: inp.deviceId);
      CudaFFIFunctions.maxpool2D(out, inp, kernelSize,
          stride: stride,
          dilation: dilation,
          padding: padding,
          padMode: padMode,
          padValue: padValue);
      return out;
    }
     */
    throw UnimplementedError();
  }

  Size2D outSize2D(Size inSize) {
    // TODO is this the right calculation?
    return inSize.twoD -
        (dilation * (kernelSize - 1)) +
        (padding * 2) ~/ stride +
        Size2D(rows: 1, cols: 1);
  }

  Size outSize(Size inSize) {
    return Size([inSize.batch, inSize.channels] + outSize2D(inSize).toList());
  }
}

// TODO free memory
/*
class Conv2D implements Layer2D {
  final Tensor _weight;

  final Tensor _bias;

  Conv2D._(this._weight, this._bias);

  factory Conv2D(int inChannels, int outChannels, int kernelSize) {
    final weight =
    Tensor.empty(Size([outChannels, inChannels, kernelSize, kernelSize]));
    final bias = Tensor.empty(Size([outChannels]));
    return Conv2D._(weight, bias);
  }

  @override
  Tensor forward(Tensor input) {
    // TODO
    throw UnimplementedError();
  }

// TODO add
}
 */