import 'dart:async';
import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U64Tensor implements Tensor<int> {
  @override
  U64Onesor get as1d;

  factory U64Tensor(U64Onesor as1d, Dim size,
          {String name = '', Context? context}) =>
      _U64Tensor(as1d, size, name: name, context: context);

  factory U64Tensor.fromList(List<int> list,
      {String name = '', Context? context, Dim? size}) {
    if (size != null) {
      if (list.length != size.nel) {
        throw ArgumentError('Size mismatch');
      }
    } else {
      size = Dim([list.length]);
    }
    U64Onesor data;
    if (tc.exists()) {
      data = U64COnesor.fromList(list, context: context);
    } else {
      data = U64DartOnesor.fromList(list);
    }
    return _U64Tensor(data, size, name: name, context: context);
  }

  factory U64Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    size = Dim.from(size);
    U64Onesor data;
    if (tc.exists()) {
      data = U64COnesor.sized(size.nel, context: context);
    } else {
      data = U64DartOnesor.sized(size.nel);
    }
    return _U64Tensor(data, size, name: name, context: context);
  }

  factory U64Tensor.generate(/* Dim | Iterable<int> | int */ size,
      int Function(Dim size, Dim index) generator,
      {String name = '', Context? context}) {
    final ret = U64Tensor.sized(size, name: name, context: context);
    for (var i = 0; i < size.nel; i++) {
      ret.as1d[i] = generator(size, size.unravel(i));
    }
    return ret;
  }

  factory U64Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    random ??= Random();
    final ret = U64Tensor.sized(size, name: name, context: context);
    for (var i = 0; i < size.nel; i++) {
      ret.as1d[i] = random.nextInt(u64.maxVal);
    }
    return ret;
  }

  @override
  Future<Tensor<int>> t({Tensor<int>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<Tensor<int>> mm(FutureOr<Tensor<int>> other, {Tensor<int>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<Tensor<int>> mmBt(FutureOr<Tensor<int>> other, {Tensor<int>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<Tensor<int>> mmColAdd(
      FutureOr<Tensor<int>> other, FutureOr<Tensor<int>> c,
      {Tensor<int>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<Tensor<int>> mmBtColAdd(
      FutureOr<Tensor<int>> other, FutureOr<Tensor<int>> c,
      {Tensor<int>? out}) {
    throw UnimplementedError();
  }
}

class _U64Tensor
    with Tensor<int>, TensorMixin<int>, U64Tensor
    implements U64Tensor, Tensor<int> {
  @override
  String name;

  @override
  final U64Onesor as1d;

  Dim _size;

  _U64Tensor(this.as1d, this._size, {this.name = 'unnamed', Context? context}) {
    context?.add(as1d);
    if (as1d.length != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
  }

  @override
  Dim get size => _size;

  @override
  set size(Dim size) {
    if (size.nel != _size.nel) {
      throw ArgumentError('Size mismatch');
    }
    _size = size;
  }
}
