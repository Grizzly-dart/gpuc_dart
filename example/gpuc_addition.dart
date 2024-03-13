import 'package:gpuc_dart/gpuc_dart.dart';

void test(Dim size) {
  final rnd = MTRandom();
  final t1 = Tensor.random(size, random: rnd);
  final t2 = Tensor.random(size, random: rnd);
  final t3 = t1 + t2;
  t3.as2d().assertEqual(t1.as2d().plus(t2.as2d()));
}

void main() async {
  initializeNativeTensorLibrary();
  test(Dim([3, 3]));
  test(Dim([512, 512]));
  test(Dim([4096, 4096]));

  print('Finished');
}
