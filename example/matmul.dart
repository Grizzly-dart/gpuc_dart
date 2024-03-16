import 'package:gpuc_dart/gpuc_dart.dart';

Future<void> main() async {
  initializeNativeTensorLibrary();
  for (int b = 1; b <= 1000; b += 7) {
    for (int m = 1; m <= 4096; m += 7) {
      for (int n = 1; n <= 4096; n += 7) {
        for (int k = 1; k <= 4096; k += 7) {
          await test(batches: b, m: m, n: n, k: k);
        }
      }
    }
  }
}

Future<void> test({int batches = 1, int m = 2, int n = 2, int k = 2}) async {
  print('=====> batches: $batches, m: $m, n: $n, k: $k');
  final a = Tensor.generate(Dim2(m, n), (i) => i.ravel + 1);
  final b = Tensor.generate(Dim2(n, k), (i) => i.ravel + 1);
  final out = await a.matmul(b);
  print(out.as1d);
  final out2 = await TensonCmd().matmul(a, b);
  out.as1d.assertEqual(out2.as1d);
}
