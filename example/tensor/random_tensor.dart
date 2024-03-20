import 'package:gpuc_dart/gpuc_dart.dart';

void main() {
  final t1 = Tensor.random([2, 3, 5]);
  print(t1[0][0]);
}