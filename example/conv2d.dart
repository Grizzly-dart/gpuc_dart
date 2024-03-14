import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/nn2d/maxpool2d.dart';

void main() {
  initializeNativeTensorLibrary();
  test1Inp1Out();
  test2Inp1Out();
  test2Inp2Out();
}

void test1Inp1Out() {
  final t1 = Tensor.fromList(
      List.generate(3 * 3, (index) => index.toDouble() + 1),
      size: Dim([3, 3]));
  final kernel = Tensor.fromList(
      List.generate(3 * 3, (index) => index.toDouble() + 1),
      size: Dim([3, 3]));
  final conv2D = Conv2D.own(kernel);
  final t2 = conv2D.forward(t1);
  print(t2.as1d);
}

void test2Inp1Out() {
  final t1 = Tensor.fromList(
      List.generate(2 * 3 * 3, (index) => index.toDouble() + 1),
      size: Dim([1, 2, 3, 3]));
  final kernel = Tensor.fromList(
      List.generate(2 * 3 * 3, (index) => index.toDouble() + 1),
      size: Dim([1, 2, 3, 3]));
  final conv2D = Conv2D.own(kernel);
  final t2 = conv2D.forward(t1);
  print(t2.as1d);
}

void test2Inp2Out() {
  final t1 = Tensor.fromList(
      List.generate(2 * 3 * 3, (index) => index.toDouble() + 1),
      size: Dim([1, 2, 3, 3]));
  final kernel = Tensor.fromList(
      List.generate(2 * 2 * 3 * 3, (index) => index.toDouble() + 1),
      size: Dim([2, 2, 3, 3]));
  final conv2D = Conv2D.own(kernel);
  final t2 = conv2D.forward(t1);
  print(t2.as1d);
}
