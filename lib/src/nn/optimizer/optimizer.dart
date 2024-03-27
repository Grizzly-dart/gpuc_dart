import 'package:gpuc_dart/gpuc_dart.dart';

abstract class Optimizer {
  Future<void> update(Tensor weights, Tensor grad);
}

class SGDSimple extends Optimizer {
  double learningRate;

  SGDSimple(this.learningRate);

  @override
  Future<void> update(Tensor weights, Tensor grad) async {
    weights -= grad * learningRate;
  }
}