import 'package:gpuc_dart/gpuc_dart.dart';

void main() async {
  initializeTensorc();
  final t1 = Tensor.random(Size.twoD(512, 512));
  // print(t1[Size([1])]);
  print(t1[Size([1])]);
  print('Finished');
}