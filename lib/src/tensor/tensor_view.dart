import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class TensorView<T extends num> implements Tensor<T> {
  Dim get offset;

  @override
  OnesorView<T> get as1d;

  @override
  void release() {}
}
