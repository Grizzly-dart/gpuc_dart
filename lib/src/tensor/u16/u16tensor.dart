import 'dart:async';
import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';

export 'u16tensor_view.dart';

abstract mixin class U16Tensor implements Tensor<int> {
  @override
  U16Onesor get as1d;

  factory U16Tensor(U16Onesor as1d, Dim size,
          {String name = '', Context? context}) =>
      _U16Tensor(as1d, size, name: name, context: context);

  factory U16Tensor.fromList(List<int> list,
      {String name = '', Context? context, Dim? size}) {
    if (size != null) {
      if (list.length != size.nel) {
        throw ArgumentError('Size mismatch');
      }
    } else {
      size = Dim([list.length]);
    }
    U16Onesor data;
    if (cffi != null) {
      data = U16COnesor.fromList(list, context: context);
    } else {
      data = U16DartOnesor.fromList(list);
    }
    return _U16Tensor(data, size, name: name, context: context);
  }

  factory U16Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    size = Dim.from(size);
    U16Onesor data;
    if (cffi != null) {
      data = U16COnesor.sized(size.nel, context: context);
    } else {
      data = U16DartOnesor.sized(size.nel);
    }
    return _U16Tensor(data, size, name: name, context: context);
  }

  factory U16Tensor.generate(/* Dim | Iterable<int> | int */ size,
      int Function(Dim size, Dim index) generator,
      {String name = '', Context? context}) {
    final ret = U16Tensor.sized(size, name: name, context: context);
    for (var i = 0; i < size.nel; i++) {
      ret.as1d[i] = generator(size, size.unravel(i));
    }
    return ret;
  }

  factory U16Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    random ??= Random();
    final ret = U16Tensor.sized(size, name: name, context: context);
    for (var i = 0; i < size.nel; i++) {
      ret.as1d[i] = random.nextInt(u16.maxVal);
    }
    return ret;
  }

  @override
  U16TensorView operator [](dynamic /* Dim | int | Iterable<int> */ index) {
    if (index is Extent) {
      if(index is! Extent<Dim>) index = Dim.extentFrom(index);
      if (index.lower.dims != index.upper.dims) {
        throw ArgumentError('Extents dimension mismatch');
      }
      if (!size.isIndex(index.upper)) {
        throw ArgumentError('Index out of range');
      }
      if (!index.lower.asList
          .take(index.lower.dims - 1)
          .isEqual(index.upper.asList.take(index.upper.dims - 1))) {
        throw ArgumentError('Extents dimension mismatch');
      }
      if (index.lower.asList.last > index.upper.asList.last) {
        throw ArgumentError(
            'Invalid Extent! lower bound is greater than upper bound');
      }

      return U16TensorView(
          this,
          index.lower,
          Dim([
            index.upper.asList.last - index.lower.asList.last + 1,
            ...size.asList.skip(index.lower.dims)
          ]));
    }

    if (index is! Dim) index = Dim.from(index);
    return U16TensorView(this, index, Dim(size.asList.skip(index.dims)));
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

class _U16Tensor with Tensor<int>, U16Tensor implements U16Tensor, Tensor<int> {
  @override
  String name;

  @override
  final U16Onesor as1d;

  Dim _size;

  _U16Tensor(this.as1d, this._size, {this.name = 'unnamed', Context? context}) {
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
