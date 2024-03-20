import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/nn2d/nn2d.dart';

Future<void> main() async {
  initializeNativeTensorLibrary();

  /*for (int b = 1; b < 20; b += 5) {
    await testConv2D(batches: b);
    for (int i = 2; i < 100; i += 20) {
      await testConv2D(inChannels: i, batches: b);
    }
    for (int i = 2; i < 100; i += 50) {
      for (int o = 2; o < 100; o += 25) {
        await testConv2D(inChannels: i, outChannels: o, batches: b);
      }
    }
    for (int groups = 2; groups < 10; groups++) {
      for (int i = 1; i < 10; i += 3) {
        for (int o = 1; o < 10; o += 3) {
          await testConv2D(
              inChannels: i * groups,
              outChannels: o * groups,
              groups: groups,
              batches: b);
        }
      }
    }
  }*/

  /*
  for(int b = 1; b < 5; b++) {
    await testConv2D(
      inputSize: Dim2(16, 16),
      padding: Dim2(2, 2),
      batches: b,
    );
    await testConv2D(
      inputSize: Dim2(16, 16),
      stride: Dim2(2, 2),
      batches: b,
    );
    await testConv2D(
      inputSize: Dim2(16, 16),
      dilation: Dim2(2, 2),
      batches: b,
    );
    await testConv2D(
      inputSize: Dim2(16, 16),
      dilation: Dim2(2, 2),
      stride: Dim2(2, 2),
      padding: Dim2(2, 2),
      batches: b,
    );
  }
   */

  for(final padMode in PadMode.values) {
    await testConv2D(
      inputSize: Dim2(16, 16),
      padding: Dim2(2, 2),
      padMode: padMode,
    );
    await testConv2D(
      inputSize: Dim2(16, 16),
      padSameSize: true,
      padMode: padMode,
    );
  }

  /*for(final padMode in PadMode.values) {

  }*/

  print('Finished');
}

Future<void> testConv2D({
  int batches = 1,
  Dim2 inputSize = const Dim2(3, 3),
  Dim2 kernelSize = const Dim2(3, 3),
  int inChannels = 1,
  int outChannels = 1,
  int groups = 1,
  Dim2 padding = const Dim2(0, 0),
  Dim2 stride = const Dim2(1, 1),
  Dim2 dilation = const Dim2(1, 1),
  PadMode padMode = PadMode.constant,
  bool padSameSize = false,
}) async {
  print(
      '=====> Batches: $batches Input: $inChannels Output: $outChannels Groups: $groups');
  final t1 = Tensor.fromList(
      List.generate(batches * inChannels * inputSize.rows * inputSize.cols,
          (index) => (index.toDouble() + 1) * 1e-6),
      size: Dim([batches, inChannels, inputSize.rows, inputSize.cols]));
  final kernel = Tensor.fromList(
      List.generate(
          outChannels *
              inChannels ~/
              groups *
              kernelSize.rows *
              kernelSize.cols,
          (index) => index.toDouble() + 1),
      size: Dim([
        outChannels,
        inChannels ~/ groups,
        kernelSize.rows,
        kernelSize.cols
      ]));
  final conv2D = Conv2D.withWeights(kernel,
      groups: groups,
      padding: padding,
      stride: stride,
      dilation: dilation,
      padMode: padMode,
      padSameSize: padSameSize);
  final t2 = await conv2D.forward(t1);
  print(t2.size);
  print(t2.as1d);
  final tensonCmd = TensonCmd();
  final resp = await tensonCmd.conv2D(
      kernel: kernel,
      input: t1,
      groups: groups,
      padding: padding,
      stride: stride,
      dilation: dilation,
      padMode: padMode,
      padSameSize: padSameSize);
  print(resp.as1d);
  t2.as1d.assertEqual(resp.as1d, eps: t1.nel * t1.nel * 1e-5);
}
