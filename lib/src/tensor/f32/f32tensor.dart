import 'dart:async';
import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class F32Tensor implements Tensor<double> {
  @override
  F32Onesor get as1d;

  factory F32Tensor(F32Onesor as1d, Dim size,
          {String name = '', Context? context}) =>
      _F32Tensor(as1d, size, name: name, context: context);

  factory F32Tensor.fromList(List<double> list,
      {String name = '', Context? context, Dim? size}) {
    if (size != null) {
      if (list.length != size.nel) {
        throw ArgumentError('Size mismatch');
      }
    } else {
      size = Dim([list.length]);
    }
    // TODO check if C/Dart
    final data = F32COnesor.fromList(list, context: context);
    return F32Tensor(data, size, name: name, context: context);
  }

  factory F32Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    return F32Tensor(F32COnesor.sized(size.nel, context: context), size,
        name: name, context: context);
  }

  factory F32Tensor.generate(/* Dim | Iterable<int> | int */ size,
      double Function(Dim size, Dim index) generator,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    final data = F32COnesor.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = generator(size, size.unravel(i));
    }
    return F32Tensor(data, size, name: name, context: context);
  }

  factory F32Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    random ??= Random();
    final data = F32COnesor.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = random.nextDouble();
    }
    return F32Tensor(data, size, name: name, context: context);
  }

  @override
  Future<Tensor<double>> t({Tensor<double>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<Tensor<double>> matmul(FutureOr<Tensor<double>> other,
      {Tensor<double>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<Tensor<double>> matmulT(FutureOr<Tensor<double>> other,
      {Tensor<double>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<Tensor<double>> matmulCadd(
      FutureOr<Tensor<double>> other, FutureOr<Tensor<double>> c,
      {Tensor<double>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<Tensor<double>> matmulCaddT(
      FutureOr<Tensor<double>> other, FutureOr<Tensor<double>> c,
      {Tensor<double>? out}) {
    throw UnimplementedError();
  }
}

class _F32Tensor
    with Tensor<double>, F32Tensor
    implements F32Tensor, Tensor<double> {
  @override
  String name;

  @override
  final F32Onesor as1d;

  Dim _size;

  _F32Tensor(this.as1d, this._size, {this.name = 'unnamed', Context? context}) {
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
