import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';

extension F64TensorFutureExt on Future<F64Tensor> {
  Future<F64Tensor> operator +(FutureOr<F64Tensor> other) async {
    final t1 = await this;
    final t2 = await other;
    return t1 + t2;
  }

  Future<F64Tensor> sumRows() async {
    final t = await this;
    return t.sumRows();
  }
}
