import 'package:gpuc_dart/gpuc_dart.dart';

void test() {
  final rnd = MTRandom();
  final t1 = Tensor.random(Dim.twoD(512, 512), random: rnd);
  final t3 = t1.sumRows();
  t1.as2d().sumRows.assertEqual(t3.as1d);
}

void main() async {
  initializeNativeTensorLibrary();
  test();
  print('Finished');
}