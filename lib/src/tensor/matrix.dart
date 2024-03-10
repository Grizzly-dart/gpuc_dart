import 'dart:collection';

import 'package:gpuc_dart/gpuc_dart.dart';

class Matrix with ListMixin<List<double>> implements List<List<double>> {
  final Tensor tensor;

  final int colDims;

  Matrix(this.tensor, {this.colDims = 1});

  late final rowDims = tensor.size.dims - colDims;

  late final Dim2 size = tensor.size.squeeze2D(colDims: colDims);

  @override
  int get length => size.rows;

  @override
  List<double> operator [](int index) =>
      tensor.as1d.view(index * size.cols, size.cols);

  @override
  void operator []=(int index, List<double> value) {
    tensor.as1d.view(index * size.cols, size.cols).copyFrom(DartList.own(value));
  }

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of matrix');
  }
}
