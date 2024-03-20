import 'dart:async';
import 'dart:math';
import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/tensor/tensor_mixin.dart';

export 'dim.dart';
export 'matrix.dart';
export 'tensor_future.dart';
export 'tensor2d_mixin.dart';
export 'tensor_view.dart';
export 'int_tensor/int_tensor.dart';

class _Tensor
    with TensorMixin, Tensor2dMixin, TypedTensorMixin<double>
    implements Tensor {
  @override
  String name;

  @override
  final Onesor<double> as1d;

  Dim _size;

  _Tensor(this.as1d, this._size, {this.name = '', Context? context}) {
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

abstract class Tensor implements TypedTensor<double> {
  factory Tensor(Onesor<double> as1d, Dim size,
          {String name = '', Context? context}) =>
      _Tensor(as1d, size, name: name, context: context);

  factory Tensor.fromList(List<double> list,
      {String name = '', Context? context, Dim? size}) {
    if (size != null) {
      if (list.length != size.nel) {
        throw ArgumentError('Size mismatch');
      }
    } else {
      size = Dim([list.length]);
    }
    final data = F64COnesor.fromList(list, context: context);
    return Tensor(data, size, name: name, context: context);
  }

  factory Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    return Tensor(F64COnesor.sized(size.nel, context: context), size,
        name: name, context: context);
  }

  factory Tensor.generate(/* Dim | Iterable<int> | int */ size,
      double Function(Dim size, Dim index) generator,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    final data = F64COnesor.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = generator(size, size.unravel(i));
    }
    return Tensor(data, size, name: name, context: context);
  }

  factory Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    random ??= Random();
    final data = F64COnesor.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = random.nextDouble();
    }
    return Tensor(data, size, name: name, context: context);
  }

  @override
  Tensor slice(/* Dim | int | Iterable<int> */ index, {Context? context});

  Tensor operator [](dynamic /* Dim | int | Iterable<int> */ index);

  void operator []=(
      dynamic /* Dim | int | Iterable<int> */ index, Tensor value);

  // TODO return NListView
  Onesor<double> row(int index, {int colDims = 1});

  @override
  Future<TypedTensor<double>> t({covariant TypedTensor<double>? out});

  @override
  Future<TypedTensor<double>> matmul(FutureOr<TypedTensor<double>> other,
      {TypedTensor<double>? out});

  @override
  Future<TypedTensor<double>> matmulT(FutureOr<TypedTensor<double>> other,
      {TypedTensor<double>? out});

  @override
  Future<TypedTensor<double>> matmulCadd(
      FutureOr<TypedTensor<double>> other, FutureOr<TypedTensor<double>> c,
      {TypedTensor<double>? out});

  @override
  Future<TypedTensor<double>> matmulCaddT(
      FutureOr<TypedTensor<double>> other, FutureOr<TypedTensor<double>> c,
      {TypedTensor<double>? out});

  Future<Tensor> operator +(covariant FutureOr<Tensor> other);

  Future<Tensor> operator -(covariant FutureOr<Tensor> other);

  Future<Tensor> operator *(covariant FutureOr<Tensor> other);

  Future<Tensor> operator /(covariant FutureOr<Tensor> other);

  Future<Tensor> sumRows({int colDims = 1});

// TODO Tensor rearrange(List<int> order, {DeviceType? forceDeviceType});
}
