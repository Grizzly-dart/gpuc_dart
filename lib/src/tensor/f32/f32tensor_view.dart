import 'package:gpuc_dart/gpuc_dart.dart';

class F32TensorView
    with Tensor<double>, F32Tensor
    implements F32Tensor, TensorView<double> {
  @override
  String name = 'unnamed';

  final F32Tensor _inner;

  @override
  final Dim offset;

  @override
  final Dim size;

  @override
  late final F32OnesorView as1d = _inner.as1d
      .view((offset.asList * _inner.size.strides.asList).sum, size.nel);

  F32TensorView(this._inner, this.offset, this.size, {this.name = 'unnamed'}) {
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
