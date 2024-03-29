import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

extension TensorInplaceExt<T extends num> on Tensor<T> {
  Future<Tensor> plus_(FutureOr<Tensor> other) => plus(other, out: this);
  Future<Tensor> sub_(FutureOr<Tensor> other) => sub(other, out: this);
  Future<Tensor> mul_(FutureOr<Tensor> other) => mul(other, out: this);
  Future<Tensor> div_(FutureOr<Tensor> other) => mul(other, out: this);

  Future<Tensor<double>> abs_() => abs(out: this) as Future<Tensor<double>>;

  Future<Tensor<T>> sqr_() => sqr(out: this) as Future<Tensor<T>>;
  Future<Tensor<T>> sqrt_() => sqrt(out: this) as Future<Tensor<T>>;
  Future<Tensor<T>> exp_() => exp(out: this) as Future<Tensor<T>>;
}
