import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

export 'groupnorm2d.dart';
export 'conv2d.dart';

abstract class Layer2D implements Layer {
  @override
  Future<Tensor> forward(FutureOr<Tensor> input);

  // TODO load wandb

  // TODO describe
}
