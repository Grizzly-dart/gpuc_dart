import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

export 'groupnorm2d.dart';
export 'conv2d.dart';

abstract class Layer2D<I extends num> implements Layer<I> {
  @override
  Future<Tensor> forward(FutureOr<TypedTensor<I>> input);

  // TODO load wandb

  // TODO describe
}
