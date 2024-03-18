import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';

Future<void> main() async {
  initializeNativeTensorLibrary();
  await test(m: 1, n: 2, k: 1);
  await test(m: 2, n: 2, k: 2);
  await test(m: 16, n: 4, k: 10);
  await test(m: 34, n: 67, k: 43);
  for(int m = 1; m <= 33; m++) {
    for(int n = 1; n <= 33; n++) {
      for(int k = 1; k <= 33; k++) {
        await test(m: m, n: n, k: k);
      }
    }
  }
  await test(batches: 1, m: 32, n: 16, k: 24);
  await test(batches: 10, m: 4096, n: 2048, k: 4096);
  await test(batches: 1, m: 4096, n: 2048, k: 4096);
  /*for (int b = 1; b <= 1000; b += 7) {
    for (int m = 1; m <= 4096; m += 7) {
      for (int n = 1; n <= 4096; n += 7) {
        for (int k = 1; k <= 4096; k += 7) {
          await test(batches: b, m: m, n: n, k: k);
        }
      }
    }
  }*/
  print('Finished!');
}

Future<void> test(
    {int batches = 1,
    int m = 2,
    int n = 2,
    int k = 2}) async {
  final rand = Random(batches * m * n * k);
  print('=====> batches: $batches, m: $m, n: $n, k: $k');
  final a = Tensor.generate(Dim2(m, n), (i) => rand.nextDouble());
  final b = Tensor.generate(Dim2(n, k), (i) => rand.nextDouble());
  // TODO transpose b
  // a.printTextTable();
  // b.printTextTable();
  final watch = Stopwatch()..start();
  final out = await a.matmulT(b);
  print('Elapsed: ${watch.elapsedMilliseconds} ms');
  print(out.as1d);
  final out2 = await TensonCmd().matmul(a, b);
  try {
    out.as1d.assertEqual(out2.as1d, eps: m * n * k * 1e-6);
  } catch (e) {
    print('a: $a');
    print('b: $b');
    rethrow;
  }
}
