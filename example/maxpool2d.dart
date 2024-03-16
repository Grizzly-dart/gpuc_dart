import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/nn2d/maxpool2d.dart';

Future<void> main() async {
  initializeNativeTensorLibrary();
  await test(kernelSize: Dim2(3, 3));
  await test(kernelSize: Dim2(3, 3), padding: Dim2(1, 1));

  print('Finished');
}

Future<void> test(
    {required Dim2 kernelSize,
    Dim2? stride,
    Dim2 padding = const Dim2(0, 0)}) async {
  final rand = MTRandom();
  final input = Tensor.fromList(
      List.generate(16 * 16, (index) => rand.nextDouble()),
      size: Dim([1, 1, 16, 16]));
  input.printTextTable();
  final maxPool2D = MaxPool2D(kernelSize, stride: stride, padding: padding);
  final out = await maxPool2D.forward(input);
  print(out.size);
  out.printTextTable();
  final tenson = TensonCmd();
  final out2 = await tenson.maxPool2D(
      kernelSize: kernelSize, input: input, stride: stride, padding: padding);
  out2.printTextTable();
  out.as1d.assertEqual(out2.as1d, eps: 1e-6);
}