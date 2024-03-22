import 'dart:async';
import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U8Tensor implements Tensor<int> {
  @override
  U8Onesor get as1d;

  factory U8Tensor(U8Onesor as1d, Dim size,
      {String name = '', Context? context}) =>
      _U8Tensor(as1d, size, name: name, context: context);

  factory U8Tensor.fromList(List<int> list,
      {String name = '', Context? context, Dim? size}) {
    if (size != null) {
      if (list.length != size.nel) {
        throw ArgumentError('Size mismatch');
      }
    } else {
      size = Dim([list.length]);
    }
    // TODO check if C/Dart
    final data = U8COnesor.fromList(list, context: context);
    return U8Tensor(data, size, name: name, context: context);
  }

  factory U8Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    return U8Tensor(U8COnesor.sized(size.nel, context: context), size,
        name: name, context: context);
  }

  factory U8Tensor.generate(/* Dim | Iterable<int> | int */ size,
      int Function(Dim size, Dim index) generator,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    final data = U8COnesor.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = generator(size, size.unravel(i));
    }
    return U8Tensor(data, size, name: name, context: context);
  }

  factory U8Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    random ??= Random();
    final data = U8COnesor.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = random.nextInt(u8.maxVal);
    }
    return U8Tensor(data, size, name: name, context: context);
  }

  @override
  Future<Tensor<int>> t({Tensor<int>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<Tensor<int>> matmul(FutureOr<Tensor<int>> other, {Tensor<int>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<Tensor<int>> matmulT(FutureOr<Tensor<int>> other, {Tensor<int>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<Tensor<int>> matmulCadd(
      FutureOr<Tensor<int>> other, FutureOr<Tensor<int>> c,
      {Tensor<int>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<Tensor<int>> matmulCaddT(
      FutureOr<Tensor<int>> other, FutureOr<Tensor<int>> c,
      {Tensor<int>? out}) {
    throw UnimplementedError();
  }
}

class _U8Tensor with Tensor<int>, U8Tensor implements U8Tensor, Tensor<int> {
  @override
  String name;

  @override
  final U8Onesor as1d;

  Dim _size;

  _U8Tensor(this.as1d, this._size, {this.name = 'unnamed', Context? context}) {
    context?.add(as1d);
    _finalizer.attach(this, as1d);
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

  static final _finalizer = Finalizer<Onesor>((Onesor l) {
    l.release();
  });
}