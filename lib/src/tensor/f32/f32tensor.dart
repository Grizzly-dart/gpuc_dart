import 'dart:async';
import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';

export 'f32tensor_view.dart';

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
    F32Onesor data;
    if (cffi != null) {
      data = F32COnesor.fromList(list, context: context);
    } else {
      data = F32DartOnesor.fromList(list);
    }
    return _F32Tensor(data, size, name: name, context: context);
  }

  factory F32Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    size = Dim.from(size);
    F32Onesor data;
    if (cffi != null) {
      data = F32COnesor.sized(size.nel, context: context);
    } else {
      data = F32DartOnesor.sized(size.nel);
    }
    return _F32Tensor(data, size, name: name, context: context);
  }

  factory F32Tensor.generate(/* Dim | Iterable<int> | int */ size,
      double Function(Dim size, Dim index) generator,
      {String name = '', Context? context}) {
    final ret = F32Tensor.sized(size, name: name, context: context);
    for (var i = 0; i < size.nel; i++) {
      ret.as1d[i] = generator(size, size.unravel(i));
    }
    return ret;
  }

  factory F32Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    final ret = F32Tensor.sized(size, name: name, context: context);
    random ??= Random();
    for (var i = 0; i < size.nel; i++) {
      ret.as1d[i] = random.nextDouble();
    }
    return ret;
  }

  @override
  F32TensorView operator [](dynamic /* Dim | int | Iterable<int> */ index) {
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

      return F32TensorView(
          this,
          index.lower,
          Dim([
            index.upper.asList.last - index.lower.asList.last + 1,
            ...size.asList.skip(index.lower.dims)
          ]));
    }

    if (index is! Dim) index = Dim.from(index);
    return F32TensorView(this, index, Dim(size.asList.skip(index.dims)));
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
