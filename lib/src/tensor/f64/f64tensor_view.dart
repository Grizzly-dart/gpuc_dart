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
  late final F64OnesorView as1d =
      _inner.as1d.view(offset.nel * size.nel, size.nel);

  @override
  late final Dim size = Dim(_inner.size.asList.skip(offset.dims));

  F64TensorView(this._inner, this.offset, {this.name = 'unnamed'}) {
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
