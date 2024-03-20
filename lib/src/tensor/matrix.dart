import 'dart:collection';

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/onesor/dart/dartonesor.dart';
import 'package:gpuc_dart/src/tensor/int_tensor/int_tensor.dart';

class Matrix<T extends num> with ListMixin<List<T>> implements List<List<T>> {
  final TypedTensor<T> tensor;

  final int colDims;

  Matrix(this.tensor, {this.colDims = 1});

  late final rowDims = tensor.size.dims - colDims;

  late final Dim2 size = tensor.size.squeeze2D(colDims: colDims);

  @override
  int get length => size.rows;

  @override
  List<T> operator [](int index) =>
      tensor.as1d.view(index * size.cols, size.cols);

  @override
  void operator []=(int index, List<T> value) {
    tensor.as1d.view(index * size.cols, size.cols).copyFrom(DartOnesor(value));
  }

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of matrix');
  }
}
