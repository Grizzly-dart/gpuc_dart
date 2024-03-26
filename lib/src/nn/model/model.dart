import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

class Model {
  final String name;

  final Layer layers;

  LossFunction lossFunction;

  Optimizer optimizer;

  Model(this.layers,
      {required this.optimizer,
      required this.lossFunction,
      this.name = 'unnamed'});

  Future<Tensor> predict(FutureOr<Tensor> input) => layers.predict(input);

  void train(Tensor input, Tensor target) {
    // TODO
  }
}
