import 'dart:convert';
import 'dart:io';

import 'package:gpuc_dart/gpuc_dart.dart';

class TensonCmd {
  Future<Tensor> transpose2D(Tensor input) async {
    final resp = await _execute('transpose2d', [
      TensonVar(name: 'input', data: input),
    ]);
    return resp['output']!.data as Tensor;
  }

  Future<Tensor> matmul(Tensor a, Tensor b, {Tensor? c}) async {
    final resp = await _execute('matmul', [
      TensonVar(name: 'inputA', data: a),
      TensonVar(name: 'inputB', data: b),
      if (c != null) TensonVar(name: 'inputC', data: c),
    ]);
    return resp['output']!.data as Tensor;
  }

  Future<Tensor> maxPool2D(
      {required Dim2 kernelSize,
      required Tensor input,
      Dim2 padding = const Dim2(0, 0),
      Dim2? stride = const Dim2(1, 1),
      Dim2 dilation = const Dim2(1, 1),
      PadMode padMode = PadMode.constant}) async {
    final resp = await _execute('maxpool2d', [
      TensonVar(name: 'padding_mode', data: mapPadMode(padMode)),
      TensonVar(name: 'stride', data: stride),
      TensonVar(name: 'padding', data: padding),
      TensonVar(name: 'dilation', data: dilation),
      TensonVar(name: 'kernelSize', data: kernelSize),
      TensonVar(name: 'input', data: input),
    ]);
    return resp['output']!.data as Tensor;
  }

  Future<Tensor> conv2D({
    required Tensor kernel,
    required Tensor input,
    int groups = 1,
    Dim2 padding = const Dim2(0, 0),
    Dim2 stride = const Dim2(1, 1),
    Dim2 dilation = const Dim2(1, 1),
    PadMode padMode = PadMode.constant,
    bool padSameSize = false,
  }) async {
    final resp = await _execute('conv2d', [
      TensonVar(name: 'padding_mode', data: mapPadMode(padMode)),
      if (!padSameSize)
        TensonVar(name: 'stride', data: stride)
      else
        TensonVar(name: 'stride', data: 1),
      if (!padSameSize)
        TensonVar(name: 'padding', data: padding)
      else
        TensonVar(name: 'padding', data: 'same'),
      if (!padSameSize)
        TensonVar(name: 'dilation', data: dilation)
      else
        TensonVar(name: 'dilation', data: 1),
      TensonVar(name: 'groups', data: groups),
      TensonVar(name: 'kernel', data: kernel),
      TensonVar(name: 'input', data: input),
    ]);
    return resp['output']!.data as Tensor;
  }

  Future<Tensor> linear(Tensor input, Tensor weight, Tensor? bias) async {
    final resp = await _execute('linear', [
      TensonVar(name: 'input', data: input),
      TensonVar(name: 'weight', data: weight),
      if (bias != null) TensonVar(name: 'bias', data: bias),
    ]);
    return resp['output']!.data as Tensor;
  }

  String mapPadMode(PadMode padMode) {
    switch (padMode) {
      case PadMode.constant:
        return 'zeros';
      case PadMode.reflect:
        return 'reflect';
      case PadMode.replicate:
        return 'replicate';
      case PadMode.circular:
        return 'circular';
    }
  }

  Future<Map<String, TensonVar>> _execute(
      String script, List<TensonVar> args) async {
    final process = await Process.start('bash', [
      '-c',
      'source ./test/python/activate && python3 ./test/python/$script.py'
    ]);
    process.stdin.write(jsonEncode(args));
    await process.stdin.close();
    final out = await process.stdout.transform(utf8.decoder).join();
    final err = await process.stderr.transform(utf8.decoder).join();
    if (err.isNotEmpty) throw Exception(err);
    final code = await process.exitCode;
    if (code != 0) throw Exception('exit code: $code');
    return parseTenson(out);
  }
}
