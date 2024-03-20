import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:math';
import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/tensor/tensor_mixin.dart';

export 'dim.dart';
export 'matrix.dart';
export 'tensor_future.dart';
export 'tensor2d_mixin.dart';
export 'tensor_view.dart';

class _Tensor with TensorMixin, Tensor2dMixin implements Tensor {
  @override
  String name;

  @override
  final NList as1d;

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

  static final _finalizer = Finalizer<NList>((NList l) {
    l.release();
  });
}

abstract class Tensor implements Resource {
  String get name;

  set name(String name);

  NList get as1d;

  Dim get size;
  set size(Dim size);

  int get nel;

  factory Tensor(NList as1d, Dim size, {String name = '', Context? context}) =>
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
    final data = CList.fromList(list, context: context);
    return Tensor(data, size, name: name, context: context);
  }

  factory Tensor.sized(/* Dim | Iterable<int> | int */ size,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    return Tensor(CList.sized(size.nel, context: context), size,
        name: name, context: context);
  }

  factory Tensor.generate(/* Dim | Iterable<int> | int */ size,
      double Function(Dim size, Dim index) generator,
      {String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    final data = CList.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = generator(size, size.unravel(i));
    }
    return Tensor(data, size, name: name, context: context);
  }

  factory Tensor.random(/* Dim | Iterable<int> | int */ size,
      {Random? random, String name = '', Context? context}) {
    if (size is! Dim) size = Dim.from(size);
    random ??= Random();
    final data = CList.sized(size.nel, context: context);
    for (var i = 0; i < size.nel; i++) {
      data[i] = random.nextDouble();
    }
    return Tensor(data, size, name: name, context: context);
  }

  ffi.Pointer<ffi.Double> get ptr;

  DeviceType get deviceType;

  int get deviceId;

  Device get device;

  double scalar([int index = 0]);

  void squeeze(int dims);

  set set(Tensor other);

  Tensor slice(/* Dim | int | Iterable<int> */ index, {Context? context});

  Tensor operator [](dynamic /* Dim | int | Iterable<int> */ index);

  void operator []=(
      dynamic /* Dim | int | Iterable<int> */ index, Tensor value);

  // TODO return NListView
  NList row(int index, {int colDims = 1});

  Matrix as2d({int colDims = 1});

  Matrix matrix(index);

  Future<Tensor> t({Tensor? out});

  Future<Tensor> matmul(FutureOr<Tensor> other, {Tensor? out});

  Future<Tensor> matmulT(FutureOr<Tensor> other, {Tensor? out});

  Future<Tensor> matmulCadd(FutureOr<Tensor> other, FutureOr<Tensor> c,
      {Tensor? out});

  Future<Tensor> matmulCaddT(FutureOr<Tensor> other, FutureOr<Tensor> c,
      {Tensor? out});

  Future<Tensor> operator +(covariant FutureOr<Tensor> other);

  Future<Tensor> operator -(covariant FutureOr<Tensor> other);

  Future<Tensor> operator *(covariant FutureOr<Tensor> other);

  Future<Tensor> operator /(covariant FutureOr<Tensor> other);

  Future<Tensor> sumRows({int colDims = 1});

  @override
  void release();

  bool isEqual(Tensor other, {double epsilon = 1e-8});

  void assertEqual(Tensor other, {double eps = 1e-8});

  @override
  String toString() => '$as1d';

  Tensor rearrange(List<int> order, {DeviceType? forceDeviceType});

  void printTextTable();

  Map<String, dynamic> toJson();
}
