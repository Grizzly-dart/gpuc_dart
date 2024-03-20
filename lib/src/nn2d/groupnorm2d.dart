import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

class GroupNorm2D extends Layer2D<double> {
  final int numGroups;
  final int numChannels;
  final bool affine;
  double eps;

  GroupNorm2D(this.numGroups, this.numChannels,
      {this.eps = 1e-5, this.affine = true}) {
    // TODO initialize weights and biases
  }

  @override
  Future<Tensor> forward(FutureOr<TypedTensor<double>> input) async {
    final inp = await input;
    throw UnimplementedError();
  }
}
