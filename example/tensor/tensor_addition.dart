import 'package:gpuc_dart/gpuc_dart.dart';

Future<void> plus(Dim size) async {
  print('=====> Test a + b $size');
  final rnd = MTRandom();
  final t1 = F64Tensor.random(size, random: rnd);
  final t2 = F64Tensor.random(size, random: rnd);
  print('running...');
  final out = await(t1 + t2);
  print('verifying...');
  out.as2d().assertEqual(t1.as2d().plus(t2.as2d()));
}

Future<void> plusScalar(Dim size) async {
  print('=====> Test scalar a + s $size');
  final rnd = MTRandom();
  final t1 = F64Tensor.random(size, random: rnd);
  print('running...');
  final out = await(t1 + 143);
  print('verifying...');
  out.as2d().assertEqual(t1.as2d().plus(143));
}

Future<void> plus3(Dim size) async {
  print('Test a + b + c $size');
  final rnd = MTRandom();
  final t1 = F64Tensor.random(size, random: rnd);
  final t2 = F64Tensor.random(size, random: rnd);
  final t3 = F64Tensor.random(size, random: rnd);
  print('running...');
  final out = await(t1 + t2 + t3);
  print('verifying...');
  out.as2d().assertEqual(t1.as2d().plus(t2.as2d()).plus(t3.as2d()));
}

void main() async {
  initializeNativeTensorLibrary();

  for(final sizes in [
    Dim([3, 3]),
    Dim([512, 512]),
    Dim([4096, 4096]),
  ]) {
    await plus(sizes);
    // TODO await plusScalar(sizes);
  }

  print('Finished!');
}
