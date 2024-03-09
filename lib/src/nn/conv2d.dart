import 'package:gpuc_dart/src/nn/nn.dart';

class Conv2D implements Layer2D {
  final Tensor _weight;

  final Tensor _bias;

  Conv2D._(this._weight, this._bias);

  factory Conv2D(int inChannels, int outChannels, int kernelSize) {
    final weight =
        Tensor.sized(Dim([outChannels, inChannels, kernelSize, kernelSize]));
    final bias = Tensor.sized(Dim([outChannels]));
    return Conv2D._(weight, bias);
  }

  @override
  Tensor forward(Tensor input) {
    // TODO
    throw UnimplementedError();
  }
}
