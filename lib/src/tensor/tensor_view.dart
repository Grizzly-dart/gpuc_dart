import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/tensor/tensor_mixin.dart';

class TensorView with TensorMixin, Tensor2dMixin implements Tensor {
  @override
  String name = 'unnamed';

  // TODO make as1d be view
  @override
  final NList as1d;

  Dim _size;

  TensorView(this.as1d, this._size, {this.name = 'unnamed'}) {
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