import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract class LossFunction {
  const factory LossFunction.mse() = MSE;

  String get name;

  double compute(Tensor target, Tensor predicted);

  Future<Tensor> derivative(
      FutureOr<Tensor> target, FutureOr<Tensor> predicted);
}

class MSE implements LossFunction {
  const MSE();

  @override
  String get name => 'Mean Squared Error';

  @override
  double compute(Tensor y, Tensor yDash) {
    // TODO implement
    throw UnimplementedError();
  }

  // TODO handle batches
  @override
  Future<Tensor> derivative(
          FutureOr<Tensor> target, FutureOr<Tensor> predicted) =>
      predicted - target;
}

class MAE implements LossFunction {
  const MAE();

  @override
  String get name => 'Mean Absolute Error';

  @override
  double compute(Tensor target, Tensor predicted) {
    // TODO implement
    throw UnimplementedError();
  }

  // TODO handle batches
  @override
  Future<Tensor> derivative(
      FutureOr<Tensor> target, FutureOr<Tensor> predicted) {
    // TODO implement
    throw UnimplementedError();
  }
}
