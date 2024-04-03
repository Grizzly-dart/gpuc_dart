import 'package:gpuc_dart/gpuc_dart.dart';

class U16TensorView
    with Tensor<int>, TensorView<int>, U16Tensor
    implements U16Tensor, TensorView<int> {
  @override
  String name = 'unnamed';

  @override
  final U16Tensor inner;

  @override
  final Dim offset;

  @override
  final Dim size;

  @override
  late final U16OnesorView as1d = inner.as1d
      .view((offset.asList * inner.size.strides.asList).sum, size.nel);

  U16TensorView(this.inner, this.offset, this.size, {this.name = 'unnamed'}) {
    if (!inner.size.isIndex(offset)) {
      throw ArgumentError('Index out of range');
    }
  }
}
