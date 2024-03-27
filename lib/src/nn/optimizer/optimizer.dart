import 'package:gpuc_dart/gpuc_dart.dart';

abstract class Optimizer {
  Future<void> update(Tensor weights, Tensor grad);
}