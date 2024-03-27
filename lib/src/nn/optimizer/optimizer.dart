import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/tensor/tensor_inplace.dart';

abstract class Optimizer {
  Future<void> update(Tensor weights, Tensor grad);
}

class SGDSimple extends Optimizer {
  double learningRate;

  SGDSimple(this.learningRate);

  @override
  Future<void> update(Tensor weights, Tensor grad) async {
    weights.sub_(grad * learningRate);
  }
}