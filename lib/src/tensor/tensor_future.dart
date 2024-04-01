import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

extension TensorFutureExt<T extends num> on FutureOr<Tensor<T>> {
  Future<Tensor<T>> neg({Tensor<T>? out}) async => (await this).neg(out: out);

  Future<Tensor<T>> operator -() async => (await this).neg();

  Future<Tensor> plus(Tensor other, {Tensor? out}) async =>
      (await this).plus(other, out: out);

  Future<Tensor> operator +(FutureOr<Tensor> other) async =>
      (await this) + other;

  Future<Tensor> minus(Tensor other, {Tensor? out}) async =>
      (await this).minus(other, out: out);

  Future<Tensor> operator -(FutureOr<Tensor> other) async =>
      (await this) - other;

  Future<Tensor> mul(Tensor other, {Tensor? out}) async =>
      (await this).mul(other, out: out);

  Future<Tensor> operator *(FutureOr<Tensor> other) async =>
      (await this) * other;

  Future<Tensor> div(Tensor other, {Tensor? out}) async =>
      (await this).div(other, out: out);

  Future<Tensor> operator /(FutureOr<Tensor> other) async =>
      (await this) / other;

  // TODO int division

  Future<Tensor<T>> abs({Tensor<T>? out}) async => (await this).abs(out: out);

  Future<Tensor> sqr({Tensor? out}) async => (await this).sqr(out: out);

  Future<Tensor> sqrt({Tensor? out}) async => (await this).sqrt(out: out);

  Future<Tensor> exp({Tensor? out}) async => (await this).exp(out: out);

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

  Future<Tensor<T>> t({Tensor<T>? out}) async => (await this).t(out: out);

  Future<Tensor<T>> mm(FutureOr<Tensor<T>> other, {Tensor<T>? out}) async =>
      (await this).mm(other, out: out);

  Future<Tensor<T>> mmBt(FutureOr<Tensor<T>> other, {Tensor<T>? out}) async =>
      (await this).mmBt(other, out: out);

  Future<Tensor<T>> mmColAdd(FutureOr<Tensor<T>> other, FutureOr<Tensor<T>> c,
          {Tensor<T>? out}) async =>
      (await this).mmColAdd(other, c, out: out);

  Future<Tensor<T>> mmBtColAdd(FutureOr<Tensor<T>> other, FutureOr<Tensor<T>> c,
          {Tensor<T>? out}) async =>
      (await this).mmBtColAdd(other, c, out: out);

  Future<double> mean() async => (await this).mean();
/*
  TODO
  Future<Tensor> sumRows() async {
    final t = await this;
    return t.sumRows();
  }*/
}
