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

  Future<void> train(Tensor input, Tensor target,
      {int? batchSize, int? stepsPerEpoch, int epochs = 1}) async {
    if (stepsPerEpoch != null) {
      if (input.size.batch % stepsPerEpoch != 0) {
        throw ArgumentError(
            'input batch size must be divisible by stepsPerEpoch');
      }
      batchSize = input.size.batch ~/ stepsPerEpoch;
    }
    if (batchSize == null) {
      throw ArgumentError('batchSize or stepsPerEpoch must be provided');
    }
    if (batchSize > input.size.batch) {
      throw ArgumentError(
          'batchSize must be less than or equal to input batch size');
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
      int step = 0;
      for (int i = 0; i < input.size.batch; i += batchSize) {
        final inputBatch = input/* TODO [Range(i, i + batchSize)]*/;
        final targetBatch = target/* TODO [Range(i, i + batchSize)]*/;
        final predicted = layers.train(inputBatch);
        final lossGrad = await lossFunction.derivative(targetBatch, predicted);
        lastLayer.backward(lossGrad, optimizer);
        // TODO trigger callbacks
        // TODO verbose mode
        step++;
      }
    }
  }

  // TODO evaluate

  Layer get lastLayer {
    Layer layer = layers;
    while (layer.next != null) {
      layer = layer.next!;
    }
    return layer;
  }
}