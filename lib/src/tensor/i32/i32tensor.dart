import 'dart:async';
import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I32Tensor implements Tensor<int> {
  @override
  I32Onesor get as1d;

  factory I32Tensor(I32Onesor as1d, Dim size,
          {String name = '', Context? context}) =>
      _I32Tensor(as1d, size, name: name, context: context);

  factory I32Tensor.fromList(List<int> list,
      {String name = '', Context? context, Dim? size}) {
    if (size != null) {
      if (list.length != size.nel) {
        throw ArgumentError('Size mismatch');
      }
    } else {
      size = Dim([list.length]);
    }
    I32Onesor data;
    if(tc.exists()) {
      data = I32COnesor.fromList(list, context: context);
    } else {
      data = I32DartOnesor.fromList(list);
    }
    return I32Tensor(data, size, name: name, context: context);
  }

  factory I32Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    size = Dim.from(size);
    I32Onesor data;
    if(tc.exists()) {
      data = I32COnesor.sized(size.nel, context: context);
    } else {
      data = I32DartOnesor.sized(size.nel);
    }
    return I32Tensor(data, Dim.from(size), name: name, context: context);
  }

  factory I32Tensor.generate(/* Dim | Iterable<int> | int */ size,
      int Function(Dim size, Dim index) generator,
      {String name = '', Context? context}) {
    final ret = I32Tensor.sized(size, name: name, context: context);
    for (var i = 0; i < size.nel; i++) {
      ret.as1d[i] = generator(size, size.unravel(i));
    }
    return ret;
  }

  factory I32Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    random ??= Random();
    final ret = I32Tensor.sized(size, name: name, context: context);
    for (var i = 0; i < size.nel; i++) {
      ret.as1d[i] = random.nextInt(i32.maxVal);
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

class _I32Tensor with Tensor<int>, TensorMixin<int>, I32Tensor implements I32Tensor, Tensor<int> {
  @override
  String name;

  @override
  final I32Onesor as1d;

  Dim _size;

  _I32Tensor(this.as1d, this._size, {this.name = 'unnamed', Context? context}) {
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
