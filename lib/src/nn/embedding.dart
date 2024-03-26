import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

class Embedding extends Layer<int> {
  final F64Tensor weight;

  // TODO padding index
  // TODO max norm

  Embedding.withWeights(this.weight);

  factory Embedding(int numEmbeddings, int embeddingDim) {
    final weights = F64Tensor.sized(Dim2(numEmbeddings, embeddingDim));
    // TODO fill with random normal
    return Embedding.withWeights(weights);
  }

  @override
  Future<Tensor<double>> compute(FutureOr<Tensor<int>> input,
      {covariant Tensor<double>? out, bool training = false}) async {
    return weight.pickRows(await input);
  }

  int get numEmbeddings => weight.size.rows;

  int get embeddingDim => weight.size.cols;
}
