import 'package:gpuc_dart/src/nn2d/nn2d.dart';

class Conv2D implements Layer2D {
  final Tensor weight;

  final Tensor bias;

  final Dim2 kernelSize;

  final Dim2 stride;

  final Dim2 padding;

  final double pad;

  final PadMode padMode;

  final Dim2 dilation;

  Conv2D.own(this.weight, this.bias,
      {required this.kernelSize,
      required this.stride,
      required this.dilation,
      required this.padding,
      required this.pad,
      required this.padMode}) {
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
    Dim2 padding = const Dim2(1, 1),
  }) {
    final weight = Tensor.sized(
        Dim([outChannels, inChannels, kernelSize.rows, kernelSize.cols]));
    final bias = Tensor.sized(Dim([outChannels]));
    return Conv2D.own(
      weight,
      bias,
      kernelSize: kernelSize,
      stride: stride,
      dilation: dilation,
      pad: pad,
      padMode: padMode,
      padding: padding,
    );
  }

  @override
  Tensor forward(Tensor input) {
    // TODO
    throw UnimplementedError();
  }

  // TODO is this correct calculation?
  Dim2 outSize2D(Dim inSize) =>
      (inSize.twoD + (padding * 2) - (dilation * (kernelSize - 1))) ~/ stride +
          Dim2(1, 1);

  Dim outSize(Dim inSize) {
    return Dim([inSize.batch, inSize.channels] + outSize2D(inSize).toList());
  }
}
