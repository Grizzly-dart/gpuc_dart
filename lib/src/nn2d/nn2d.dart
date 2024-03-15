import 'package:gpuc_dart/gpuc_dart.dart';

export 'package:gpuc_dart/gpuc_dart.dart';
export 'groupnorm2d.dart';
export 'conv2d.dart';

abstract class Layer2D {
  Future<Tensor> forward(Tensor input);

  // TODO load wandb

  // TODO describe
}
