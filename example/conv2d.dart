import 'dart:convert';
import 'dart:io';

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/tensor/tensor_json.dart';

Future<void> main() async {
  initializeNativeTensorLibrary();

  await testConv2D();
  await testConv2D(inChannels: 2);
  await testConv2D(inChannels: 2, outChannels: 2);
  await testConv2D(inChannels: 2, outChannels: 2, groups: 2);

  // Batches
  await testConv2D(batches: 2);
  await testConv2D(inChannels: 2, batches: 2);
  await testConv2D(inChannels: 2, outChannels: 2, batches: 2);
  await testConv2D(inChannels: 2, outChannels: 2, groups: 2, batches: 2);
}

Future<void> testConv2D(
    {int batches = 1,
    int inChannels = 1,
    int outChannels = 1,
    int groups = 1}) async {
  print(
      '=====> $batches Batches $inChannels Input $outChannels Output $groups Groups');
  final t1 = Tensor.fromList(
      List.generate(
          batches * inChannels * 3 * 3, (index) => index.toDouble() + 1),
      size: Dim([batches, inChannels, 3, 3]));
  final kernel = Tensor.fromList(
      List.generate(outChannels * inChannels ~/ groups * 3 * 3,
          (index) => index.toDouble() + 1),
      size: Dim([outChannels, inChannels ~/ groups, 3, 3]));
  final conv2D = Conv2D.own(kernel, groups: groups);
  final t2 = conv2D.forward(t1);
  print(t2.as1d);
  final resp = await python(kernel: kernel, input: t1, groups: groups);
  t2.as1d.assertEqual(resp.as1d);
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
