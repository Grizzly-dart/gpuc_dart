import 'package:gpuc_dart/gpuc_dart.dart';

void test() {
  final t1 = Tensor.random(Dim.twoD(512, 512));
  final t3 = t1.sumRows();
  print(t1[0].data.sum);
  print(t3[0].scalar());
}

void main() async {
  initializeTensorc();
  test();
  print('Finished');
}
