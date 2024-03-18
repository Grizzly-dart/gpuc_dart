import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract class Layer {
  Future<Tensor> forward(FutureOr<Tensor> input);
}

// TODO error messages should be more technical and domain specific
class Linear extends Layer {
  late Tensor weight;
  late Tensor? bias;

  Linear.withWeights(this.weight, {this.bias}) {
    if (weight.size.dims != 2) {
      throw ArgumentError('weight must be 2D');
    }
    if (bias != null) {
      if (bias!.size.dims != 1) {
        throw ArgumentError('bias must be 1D');
      }
      if (bias!.size.cols != weight.size.cols) {
        throw ArgumentError('bias columns must be equal to weight columns');
      }
    }
  }

  Linear(int inFeatures, int outFeatures, {bool bias = true}) {
    // TODO fill with random normal
    weight = Tensor.generate(Dim2(inFeatures, outFeatures), (i) => 0);
    if (bias) {
      this.bias = Tensor.sized(Dim([outFeatures]));
    } else {
      this.bias = null;
    }
  }

  @override
  Future<Tensor> forward(FutureOr<Tensor> input) async {
    final inp = await input;
    if (inp.size.cols != weight.size.rows) {
      throw ArgumentError('input columns must be equal to weight rows');
    }

    if(bias == null) {
      return inp.matmul(weight);
    } else {
      throw UnimplementedError();
    }
  }

  Dim outSize(Dim inSize) =>
      Dim([...inSize.asList.take(inSize.dims - 1), weight.size.cols]);
}
