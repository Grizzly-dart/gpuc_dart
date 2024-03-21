import 'dart:async';
import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I16Tensor implements Tensor<int> {
  @override
  I16Onesor get as1d;

  factory I16Tensor(I16Onesor as1d, Dim size,
      {String name = '', Context? context}) =>
      _I16Tensor(as1d, size, name: name, context: context);

  factory I16Tensor.fromList(List<int> list,
      {String name = '', Context? context, Dim? size}) {
    if (size != null) {
      if (list.length != size.nel) {
        throw ArgumentError('Size mismatch');
      }
    } else {
      size = Dim([list.length]);
    }
    // TODO check if C/Dart
    final data = I16COnesor.fromList(list, context: context);
    return I16Tensor(data, size, name: name, context: context);
  }

  factory I16Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    return I16Tensor(I16COnesor.sized(size.nel, context: context), size,
        name: name, context: context);
  }

  factory I16Tensor.generate(/* Dim | Iterable<int> | int */ size,
      int Function(Dim size, Dim index) generator,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    final data = I16COnesor.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = generator(size, size.unravel(i));
    }
    return I16Tensor(data, size, name: name, context: context);
  }

  factory I16Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    random ??= Random();
    final data = I16COnesor.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = random.nextInt(i16.maxVal);
    }
    return I16Tensor(data, size, name: name, context: context);
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

class _I16Tensor with Tensor<int>, I16Tensor implements I16Tensor, Tensor<int> {
  @override
  String name;

  @override
  final I16Onesor as1d;

  Dim _size;

  _I16Tensor(this.as1d, this._size, {this.name = 'unnamed', Context? context}) {
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