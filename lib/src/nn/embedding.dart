import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

class Embedding implements Layer {
  final Tensor weight;

  // TODO padding index
  // TODO max norm

  Embedding.withWeights(this.weight);

  factory Embedding(int numEmbeddings, int embeddingDim) {
    final weights = Tensor.sized(Dim2(numEmbeddings, embeddingDim));
    // TODO fill with random normal
    return Embedding.withWeights(weights);
  }

  @override
  Future<Tensor> forward(FutureOr<Tensor> input) {

    // TODO
  }

  int get numEmbeddings => weight.size.rows;
  int get embeddingDim => weight.size.cols;
}