import 'package:gpuc_dart/gpuc_dart.dart';

class U16TensorView
    with Tensor<int>, U16Tensor
    implements U16Tensor, TensorView<int> {
  @override
  String name = 'unnamed';

  final U16Tensor _inner;

  @override
  final Dim offset;

  @override
  final Dim size;

  @override
  late final U16OnesorView as1d = _inner.as1d
      .view((offset.asList * _inner.size.strides.asList).sum, size.nel);

  U16TensorView(this._inner, this.offset, this.size, {this.name = 'unnamed'}) {
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
