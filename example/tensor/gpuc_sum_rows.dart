import 'package:gpuc_dart/gpuc_dart.dart';

Future<void> test() async {
  final rnd = MTRandom();
  final t1 = Tensor.random(Dim.to2D(512, 512), random: rnd);
  final t3 = await t1.sumRows();
  t1.as2d().sumRows.assertEqual(t3.as1d);
}

Future<void> main() async {
  initializeNativeTensorLibrary();
  await test();
  print('Finished');
}