import 'dart:async';
import 'dart:math';
import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/tensor/f64/tensor_mixin.dart';

class _F64Tensor
    with Tensor<double>, F64TensorMixin, F64Tensor2dMixin
    implements F64Tensor {
  @override
  String name;

  @override
  final Onesor<double> as1d;

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

abstract class F64Tensor implements Tensor<double> {
  factory F64Tensor(Onesor<double> as1d, Dim size,
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
    final data = F64COnesor.fromList(list, context: context);
    return F64Tensor(data, size, name: name, context: context);
  }

  factory F64Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    return F64Tensor(F64COnesor.sized(size.nel, context: context), size,
        name: name, context: context);
  }

  factory F64Tensor.generate(/* Dim | Iterable<int> | int */ size,
      double Function(Dim size, Dim index) generator,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    final data = F64COnesor.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = generator(size, size.unravel(i));
    }
    return F64Tensor(data, size, name: name, context: context);
  }

  factory F64Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    random ??= Random();
    final data = F64COnesor.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = random.nextDouble();
    }
    return F64Tensor(data, size, name: name, context: context);
  }

  F64Tensor operator [](dynamic /* Dim | int | Iterable<int> */ index);

  void operator []=(
      dynamic /* Dim | int | Iterable<int> */ index, F64Tensor value);

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

  Future<F64Tensor> sumRows({int colDims = 1});

// TODO Tensor rearrange(List<int> order, {DeviceType? forceDeviceType});
}
