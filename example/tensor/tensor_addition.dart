import 'package:gpuc_dart/gpuc_dart.dart';

Future<void> test(Dim size) async {
  print('=====> Test a + b $size');
  final rnd = MTRandom();
  final t1 = F64Tensor.random(size, random: rnd);
  final t2 = F64Tensor.random(size, random: rnd);
  print('running...');
  final out = await(t1 + t2);
  print('verifying...');
  out.as2d().assertEqual(t1.as2d().plus(t2.as2d()));
}

Future<void> test3(Dim size) async {
  print('Test a + b + c $size');
  final rnd = MTRandom();
  final t1 = F64Tensor.random(size, random: rnd);
  final t2 = F64Tensor.random(size, random: rnd);
  final t3 = F64Tensor.random(size, random: rnd);
  final out = await(t1 + t2 + t3);
  print('verifying!');
  out.as2d().assertEqual(t1.as2d().plus(t2.as2d()).plus(t3.as2d()));
}

void main() async {
  initializeNativeTensorLibrary();
  await test(Dim([3, 3]));
  await test(Dim([512, 512]));
  await test(Dim([4096, 4096]));

  print('Finished!');
}
