import 'dart:async';
import 'dart:math';
import 'package:gpuc_dart/gpuc_dart.dart';

export 'f64tensor_view.dart';

abstract mixin class F64Tensor implements Tensor<double> {
  @override
  F64Onesor get as1d;

  factory F64Tensor(F64Onesor as1d, Dim size,
      {String name = '', Context? context}) =>
      _F64Tensor(as1d, size, name: name, context: context);

  factory F64Tensor.fromList(List<double> list,
      {String name = '', Context? context, Dim? size}) {
    if (size != null) {
      if (list.length != size.nel) {
        throw ArgumentError('Size mismatch');
      }
    } else {
      size = Dim([list.length]);
    }
    F64Onesor data;
    if (cffi != null) {
      data = F64COnesor.fromList(list, context: context);
    } else {
      data = F64DartOnesor.fromList(list);
    }
    return _F64Tensor(data, size, name: name, context: context);
  }

  factory F64Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    size = Dim.from(size);
    F64Onesor data;
    if (cffi != null) {
      data = F64COnesor.sized(size.nel, context: context);
    } else {
      data = F64DartOnesor.sized(size.nel);
    }
    return _F64Tensor(data, size, name: name, context: context);
  }

  factory F64Tensor.generate(/* Dim | Iterable<int> | int */ size,
      double Function(Dim size, Dim index) generator,
      {String name = '', Context? context}) {
    final ret = F64Tensor.sized(size, name: name, context: context);
    for (var i = 0; i < size.nel; i++) {
      ret.as1d[i] = generator(size, size.unravel(i));
    }
    return ret;
  }

  factory F64Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    final ret = F64Tensor.sized(size, name: name, context: context);
    random ??= Random();
    for (var i = 0; i < size.nel; i++) {
      ret.as1d[i] = random.nextDouble();
    }
    return ret;
  }

  @override
  F64TensorView operator [](dynamic /* Dim | int | Iterable<int> */ index) {
    if (index is! Dim) index = Dim.from(index);
    return F64TensorView(this, index);
  }

  // TODO return NListView
  Onesor<double> row(int index, {int colDims = 1});

  @override
  Future<Tensor<double>> t({covariant Tensor<double>? out});

  @override
  Future<Tensor<double>> matmul(FutureOr<Tensor<double>> other,
      {Tensor<double>? out});

  @override
  Future<Tensor<double>> matmulT(FutureOr<Tensor<double>> other,
      {Tensor<double>? out});

  @override
  Future<Tensor<double>> matmulCadd(
      FutureOr<Tensor<double>> other, FutureOr<Tensor<double>> c,
      {Tensor<double>? out});

  @override
  Future<Tensor<double>> matmulCaddT(
      FutureOr<Tensor<double>> other, FutureOr<Tensor<double>> c,
      {Tensor<double>? out});

// TODO Tensor rearrange(List<int> order, {DeviceType? forceDeviceType});

  // TODO start and length
  /*F64Tensor slice(/* Dim | int | Iterable<int> */ index, {Context? context}) {
    if (index is! Dim) index = Dim.from(index);
    if (size.isIndex(index)) {
      throw ArgumentError('Index out of range');
    }

    final outSize = Dim(size.asList.skip(index.dims));
    return F64Tensor(as1d.slice(index.nel * outSize.nel, outSize.nel), outSize,
        context: context);
  }*/
}

class _F64Tensor
    with Tensor<double>, F64Tensor, F64Tensor2dMixin
    implements F64Tensor {
  @override
  String name;

  @override
  final F64Onesor as1d;

  Dim _size;

  _F64Tensor(this.as1d, this._size, {this.name = '', Context? context}) {
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
