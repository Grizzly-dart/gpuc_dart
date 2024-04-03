import 'package:gpuc_dart/gpuc_dart.dart';

class F32TensorView
    with Tensor<double>, TensorView<double>, F32Tensor
    implements F32Tensor, TensorView<double> {
  @override
  String name = 'unnamed';

  @override
  final F32Tensor inner;

  @override
  final Dim offset;

  @override
  final Dim size;

  @override
  late final F32OnesorView as1d = inner.as1d
      .view((offset.asList * inner.size.strides.asList).sum, size.nel);

  F32TensorView(this.inner, this.offset, this.size, {this.name = 'unnamed'}) {
    if (!inner.size.isIndex(offset)) {
      throw ArgumentError('Index out of range');
    }
  }
}
