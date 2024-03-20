import 'package:gpuc_dart/gpuc_dart.dart';

Future<void> main() async {
  initializeNativeTensorLibrary();
  await test(10, 5, batches: Dim([100]));
  print('Finished!');
}

Future<void> test(int inFeatures, int outFeatures,
    {Dim? batches, bool withBias = false}) async {
  print(
      '=====> inFeatures: $inFeatures, outFeatures: $outFeatures, batches: $batches');
  batches ??= Dim([1]);
  final rnd = MTRandom();
  final inp = F64Tensor.random(Dim([...batches.asList, inFeatures]), random: rnd);
  final weights = F64Tensor.random(Dim([inFeatures, outFeatures]), random: rnd);
  F64Tensor? bias;
  if (withBias) {
    bias = F64Tensor.random(Dim([outFeatures]), random: rnd);
  }
  final linear = Linear.withWeights(weights, bias: bias);
  final out = await linear.forward(inp);
  final out2 = await TensonCmd().linear(inp, await weights.t(), bias);
  out.assertEqual(out2, eps: inp.size.to2D().nel * 1e-6);
}
