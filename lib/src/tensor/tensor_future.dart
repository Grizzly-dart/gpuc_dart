import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

extension TensorFutureExt<T extends num> on FutureOr<Tensor<T>> {
  Future<Tensor> operator +(FutureOr<Tensor> other) async =>
      (await this) + other;

  Future<Tensor> operator -(FutureOr<Tensor> other) async =>
      (await this) - other;

  Future<Tensor> operator *(FutureOr<Tensor> other) async =>
      (await this) * other;

  Future<Tensor> operator /(FutureOr<Tensor> other) async =>
      (await this) / other;

  Future<Tensor<double>> sin({Tensor<double>? out}) async =>
      (await this).sin(out: out);

  Future<Tensor<double>> cos({Tensor<double>? out}) async =>
      (await this).cos(out: out);

  Future<Tensor<double>> tan({Tensor<double>? out}) async => (await this).tan();

  Future<Tensor<double>> sinh({Tensor<double>? out}) async =>
      (await this).sinh();

  Future<Tensor<double>> cosh({Tensor<double>? out}) async =>
      (await this).cosh();

  Future<Tensor<double>> tanh({Tensor<double>? out}) async =>
      (await this).tanh();

/*
  TODO
  Future<Tensor> sumRows() async {
    final t = await this;
    return t.sumRows();
  }*/
}
