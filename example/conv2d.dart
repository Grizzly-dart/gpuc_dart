import 'dart:convert';
import 'dart:io';

import 'package:gpuc_dart/gpuc_dart.dart';

Future<void> main() async {
  initializeNativeTensorLibrary();

  for (int b = 1; b < 20; b += 5) {
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
  }

  print('Finished');
}

Future<void> testConv2D(
    {int batches = 1,
    int inChannels = 1,
    int outChannels = 1,
    int groups = 1}) async {
  print(
      '=====> Batches: $batches Input: $inChannels Output: $outChannels Groups: $groups');
  final t1 = Tensor.fromList(
      List.generate(batches * inChannels * 3 * 3,
          (index) => (index.toDouble() + 1) * 1e-6),
      size: Dim([batches, inChannels, 3, 3]));
  final kernel = Tensor.fromList(
      List.generate(outChannels * inChannels ~/ groups * 3 * 3,
          (index) => index.toDouble() + 1),
      size: Dim([outChannels, inChannels ~/ groups, 3, 3]));
  final conv2D = Conv2D.own(kernel, groups: groups);
  final t2 = await conv2D.forward(t1);
  print(t2.as1d);
  final resp = await python(kernel: kernel, input: t1, groups: groups);
  print(resp.as1d);
  t2.as1d.assertEqual(resp.as1d, eps: t1.nel * t1.nel * 1e-5);
}

Future<Tensor> python(
    {required Tensor kernel, required Tensor input, int groups = 1}) async {
  final process = await Process.start('bash', [
    '-c',
    'source ./test/python/activate && python3 ./test/python/conv2d.py'
  ]);
  process.stdin.write(jsonEncode([
    TensonVar(name: 'groups', data: groups),
    TensonVar(name: 'kernel', data: kernel),
    TensonVar(name: 'input', data: input),
  ]));
  await process.stdin.close();
  final out = await process.stdout.transform(utf8.decoder).join();
  final err = await process.stderr.transform(utf8.decoder).join();
  if (err.isNotEmpty) throw Exception(err);
  final code = await process.exitCode;
  if (code != 0) throw Exception('exit code: $code');
  final resp = parseTenson(out);
  return resp['output']!.data as Tensor;
}
