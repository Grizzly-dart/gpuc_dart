import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

export 'activation/activation.dart';
export 'loss_function/loss_function.dart';
export 'optimizer/optimizer.dart';
export 'model/model.dart';
export 'normalization/normalization.dart';

export 'nn2d/nn2d.dart';

abstract class Layer<I extends num> {
  Future<Tensor> compute(FutureOr<Tensor<I>> input,
      {Tensor? out, bool training = false});

  Future<Tensor> predict(FutureOr<Tensor<I>> input) async {
    final output = await compute(input);
    if (next != null) {
      return next!.predict(output);
    }
    return output;
  }

  Future<Tensor> train(FutureOr<Tensor<I>> input, BackPropagator bp) async {
    final inp = await input;
    final output = await compute(inp, training: true);

    Tensor djByDy;
    if (next == null) {
      bp.predicted = output;
      djByDy = await bp.lf.derivative(bp.target, output);
    } else {
      djByDy = await next!.train(output, bp);
    }

    return computeBackward(inp, djByDy, bp.optimizer);
  }

  Future<Tensor> computeBackward(
      Tensor input, Tensor djByDy, Optimizer optimizer);

  Layer? prev;
  Layer? next;

  Layer chain(Layer next) {
    this.next = next;
    next.prev = this;
    return next;
  }
}

// TODO error messages should be more technical and domain specific
class Linear extends Layer<double> {
  late F64Tensor weight;
  late F64Tensor? bias;

  Linear.withWeights(this.weight, {this.bias}) {
    if (weight.size.dims != 2) {
      throw ArgumentError('weight must be 2D');
    }
    if (bias != null) {
      if (bias!.size.cols != weight.size.cols) {
        throw ArgumentError('bias columns must be equal to weight columns');
      }
      if (bias!.nel != weight.size.cols) {
        throw ArgumentError('bias nel must be equal to weight columns');
      }
    }
  }

  Linear(int inFeatures, int outFeatures, {bool bias = true}) {
    // TODO fill with random normal
    weight = F64Tensor.generate(Dim2(inFeatures, outFeatures), (_, v) => 0);
    if (bias) {
      this.bias = F64Tensor.sized(Dim([outFeatures]));
    } else {
      this.bias = null;
    }
  }

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor<double>> input,
      {Tensor? out, bool training = false}) async {
    final inp = await input;
    if (inp.size.cols != weight.size.rows) {
      throw ArgumentError('input columns must be equal to weight rows');
    }

    if (bias == null) {
      return inp.mm(weight);
    } else {
      return inp.mmColAdd(weight, bias!);
    }
  }

  Dim outSize(Dim inSize) =>
      Dim([...inSize.asList.take(inSize.dims - 1), weight.size.cols]);

  @override
  Future<Tensor> computeBackward(
      Tensor input, Tensor djByDy, Optimizer optimizer) async {
    final djByDw = await input.mmAt(djByDy);
    await optimizer.update(weight, djByDw);
    if (bias != null) {
      final djByDb = djByDy;
      await optimizer.update(bias!, djByDb);
    }

    final djByDx = djByDy.mmBt(weight);
    return djByDx;
  }
}
