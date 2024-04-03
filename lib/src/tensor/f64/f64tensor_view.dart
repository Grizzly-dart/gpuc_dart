import 'package:gpuc_dart/gpuc_dart.dart';

class F64TensorView
    with Tensor<double>, TensorView<double>, F64Tensor, F64Tensor2dMixin
    implements F64Tensor, TensorView<double> {
  @override
  String name = 'unnamed';

  @override
  final F64Tensor inner;

  @override
  final Dim offset;

  @override
  final Dim size;

  @override
  late final F64OnesorView as1d = inner.as1d
      .view((offset.asList * inner.size.strides.asList).sum, size.nel);

  F64TensorView(this.inner, this.offset, this.size, {this.name = 'unnamed'}) {
    if (!inner.size.isIndex(offset)) {
      throw ArgumentError('Index out of range');
    }
  }
}
