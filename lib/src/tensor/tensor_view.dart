import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/tensor/int_tensor/int_tensor.dart';
import 'package:gpuc_dart/src/tensor/tensor_mixin.dart';

class OffsetF64TensorView
    with F64TensorMixin, F64Tensor2dMixin, TypedTensorMixin<double>
    implements F64Tensor {
  @override
  String name = 'unnamed';

  final F64Tensor _inner;

  final Dim offset;

  @override
  late final OnesorView<double> as1d =
      _inner.as1d.view(offset.nel * size.nel, size.nel);

  late Dim _size = Dim(size.asList.skip(offset.dims));

  OffsetF64TensorView(this._inner, this.offset, {this.name = 'unnamed'}) {
    // TODO validate
    if (as1d.length != size.nel) {
      throw ArgumentError('Size does not match');
    }
  }

  @override
  Dim get size => _size;

  @override
  set size(Dim newSize) {
    if (newSize.nel != nel) {
      throw ArgumentError('Size does not match');
    }
    _size = newSize;
  }

  @override
  void release() {}
}
