import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

class GroupNorm extends Layer2D<double> {
  final int numGroups;
  final int numChannels;
  double eps;
  final bool affine;

  GroupNorm(this.numGroups, this.numChannels,
      {this.eps = 1e-5, this.affine = true}) {
    // TODO initialize weights and biases
  }

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor<double>> input,
      {covariant Tensor<double>? out, bool training = false}) async {
    final inp = await input;
    throw UnimplementedError();
  }

  @override
  Future<Tensor> computeBackward(
      Tensor input, Tensor djByDy, Optimizer optimizer) {
    // TODO
    throw UnimplementedError();
  }
}
