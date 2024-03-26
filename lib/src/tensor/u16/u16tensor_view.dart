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
  late final U16OnesorView as1d =
      _inner.as1d.view(offset.nel * size.nel, size.nel);

  @override
  late final Dim size = Dim(_inner.size.asList.skip(offset.dims));

  U16TensorView(this._inner, this.offset, {this.name = 'unnamed'}) {
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
