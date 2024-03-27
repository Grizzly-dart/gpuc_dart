import 'package:gpuc_dart/gpuc_dart.dart';

class F64TensorView
    with Tensor<double>, F64Tensor, F64Tensor2dMixin
    implements F64Tensor, TensorView<double> {
  @override
  String name = 'unnamed';

  final F64Tensor _inner;

  @override
  final Dim offset;

  @override
  final Dim size;

  @override
  late final F64OnesorView as1d = _inner.as1d
      .view((offset.asList * _inner.size.strides.asList).sum, size.nel);

  F64TensorView(this._inner, this.offset, this.size, {this.name = 'unnamed'}) {
    if (!_inner.size.isIndex(offset)) {
      throw ArgumentError('Index out of range');
    }
  }

  @override
  set size(Dim newSize) {
    throw UnsupportedError('Cannot set size of view');
  }

  @override
  void release() {}
}
