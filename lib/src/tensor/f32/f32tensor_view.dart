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
  late final F32OnesorView as1d =
      _inner.as1d.view(offset.nel * size.nel, size.nel);

  @override
  late final Dim size = Dim(_inner.size.asList.skip(offset.dims));

  F32TensorView(this._inner, this.offset, {this.name = 'unnamed'}) {
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
