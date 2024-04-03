import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class TensorView<T extends num> implements Tensor<T> {
  Tensor<T> get inner;

  Dim get offset;

  @override
  set size(Dim newSize) {
    throw UnsupportedError('Cannot set size of view');
  }

  @override
  void release() {}

  @override
  void coRelease(Resource other) {
    inner.coRelease(other);
  }

  @override
  void detachCoRelease(Resource other) {
    inner.detachCoRelease(other);
  }
}
