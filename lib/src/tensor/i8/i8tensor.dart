import 'dart:async';
import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I8Tensor implements Tensor<int> {
  @override
  I8Onesor get as1d;

  factory I8Tensor(I8Onesor as1d, Dim size,
      {String name = '', Context? context}) =>
      _I8Tensor(as1d, size, name: name, context: context);

  factory I8Tensor.fromList(List<int> list,
      {String name = '', Context? context, Dim? size}) {
    if (size != null) {
      if (list.length != size.nel) {
        throw ArgumentError('Size mismatch');
      }
    } else {
      size = Dim([list.length]);
    }
    I8Onesor data;
    if (cffi != null) {
      data = I8COnesor.fromList(list, context: context);
    } else {
      data = I8DartOnesor.fromList(list);
    }
    return _I8Tensor(data, size, name: name, context: context);
  }

  factory I8Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    size = Dim.from(size);
    I8Onesor data;
    if (cffi != null) {
      data = I8COnesor.sized(size.nel, context: context);
    } else {
      data = I8DartOnesor.sized(size.nel);
    }
    return _I8Tensor(data, size, name: name, context: context);
  }

  factory I8Tensor.generate(/* Dim | Iterable<int> | int */ size,
      int Function(Dim size, Dim index) generator,
      {String name = '', Context? context}) {
    final ret = I8Tensor.sized(size, name: name, context: context);
    for (var i = 0; i < size.nel; i++) {
      ret.as1d[i] = generator(size, size.unravel(i));
    }
    return ret;
  }

  factory I8Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    random ??= Random();
    final ret = I8Tensor.sized(size, name: name, context: context);
    for (var i = 0; i < size.nel; i++) {
      ret.as1d[i] = random.nextInt(i8.maxVal);
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

class _I8Tensor with Tensor<int>, I8Tensor implements I8Tensor, Tensor<int> {
  @override
  String name;

  @override
  final I8Onesor as1d;

  Dim _size;

  _I8Tensor(this.as1d, this._size, {this.name = 'unnamed', Context? context}) {
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

  @override
  void release() {
    as1d.release();
  }
}