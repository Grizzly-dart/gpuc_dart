import 'package:gpuc_dart/gpuc_dart.dart';

void main() {
  initializeNativeTensorLibrary();
  test1Inp1Out();
  test2Inp1Out();
  test2Inp2Out();
  test2Inp2Out2Groups();
}

void test1Inp1Out() {
  print('=====> 1 Input 1 Output');
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
  print('=====> 2 Input 1 Output');
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
  print('=====> 2 Input 2 Output');
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

void test2Inp2Out2Groups() {
  print('=====> 2 Input 2 Output 2 Groups');
  final t1 = Tensor.fromList(
      List.generate(2 * 3 * 3, (index) => index.toDouble() + 1),
      size: Dim([1, 2, 3, 3]));
  final kernel = Tensor.fromList(
      List.generate(2 * 1 * 3 * 3, (index) => index.toDouble() + 1),
      size: Dim([2, 1, 3, 3]));
  final conv2D = Conv2D.own(kernel, groups: 2);
  final t2 = conv2D.forward(t1);
  print(t2.as1d);
}
