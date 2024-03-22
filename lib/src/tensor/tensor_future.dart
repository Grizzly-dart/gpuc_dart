import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

extension TensorFutureExt<T extends num> on Future<Tensor<T>> {
  Future<Tensor> operator +(FutureOr<Tensor> other) async {
    final t1 = await this;
    final t2 = await other;
    return t1 + t2;
  }

  /*
  TODO
  Future<Tensor> sumRows() async {
    final t = await this;
    return t.sumRows();
  }*/
}
