import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

class GroupNorm extends Layer<double> {
  final int numGroups;
  final int numChannels;
  double eps;
  final bool affine;

  late final Tensor<double>? gamma;
  late final Tensor<double>? beta;

  GroupNorm(this.numGroups, this.numChannels,
      {this.eps = 1e-5, this.affine = true}) {
    assert(numChannels % numGroups == 0);
    if(affine) {
      gamma = F64Tensor.generate(numChannels, (size, index) => 1);
      beta = F64Tensor.generate(numChannels, (size, index) => 0);
    } else {
      gamma = null;
      beta = null;
    }
  }

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor<double>> input,
      {covariant Tensor<double>? out, bool training = false}) async {
    final inp = await input;
    assert(inp.size.dims > 2);
    assert(inp.size[1] == numChannels);

    throw UnimplementedError();
  }

  @override
  Future<Tensor> computeBackward(
      Tensor input, Tensor djByDy, Optimizer optimizer) {
    // TODO
    throw UnimplementedError();
  }
}
