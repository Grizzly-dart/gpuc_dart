import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';

Future<void> main() async {
  initializeNativeTensorLibrary();
  await test(Dim([2, 3]));
  await test(Dim([36, 36]));
  await test(Dim([4098, 8057]));
  print('Finished!');
}

Future<void> test(Dim size) async {
  print('=====> size: $size');
  final rand = Random(size.nel);
  final t = Tensor.generate(
      size, (s, i) => rand.nextDouble() /*size.id2D(i).toDouble()*/);
  final out = await t.t();
  final out2 = await TensonCmd().transpose2D(t);
  out.assertEqual(out2, eps: 1e-4);
}
