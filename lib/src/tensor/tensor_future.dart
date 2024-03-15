import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

extension TensorFutureExt on Future<Tensor> {
  Future<Tensor> operator +(FutureOr<Tensor> other) async {
    final t1 = await this;
    final t2 = await other;
    return t1 + t2;
  }

  Future<Tensor> sumRows() async {
    final t = await this;
    return t.sumRows();
  }
}
