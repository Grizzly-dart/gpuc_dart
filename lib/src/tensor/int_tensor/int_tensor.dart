import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:text_table/text_table.dart';

abstract class TypedTensor<T extends num> implements Resource {
  String get name;

  set name(String name);

  Onesor<T> get as1d;

  Dim get size;

  set size(Dim size);

  int get nel;

  DeviceType get deviceType;

  int get deviceId;

  Device get device;

  T scalar([int index = 0]);

  void squeeze(int dims);

  set set(TypedTensor<T> other);

  Matrix<T> matrix(int index);

  TypedTensor<T> slice(/* Dim | int | Iterable<int> */ index,
      {Context? context});

  bool isEqual(TypedTensor<T> other, {double epsilon = 1e-8});

  void assertEqual(TypedTensor<T> other, {double eps = 1e-8});

  Matrix<T> as2d({int colDims = 1});

  Future<TypedTensor<T>> t({TypedTensor<T>? out});

  Future<TypedTensor<T>> matmul(FutureOr<TypedTensor<T>> other,
      {TypedTensor<T>? out});

  Future<TypedTensor<T>> matmulT(FutureOr<TypedTensor<T>> other,
      {TypedTensor<T>? out});

  Future<TypedTensor<T>> matmulCadd(
      FutureOr<TypedTensor<T>> other, FutureOr<TypedTensor<T>> c,
      {TypedTensor<T>? out});

  Future<TypedTensor<T>> matmulCaddT(
      FutureOr<TypedTensor<T>> other, FutureOr<TypedTensor<T>> c,
      {TypedTensor<T>? out});

  void printTextTable(
      {int precision = 4 /* TODO , int? tableWidth, int? maxChars*/
      });

  Map<String, dynamic> toJson();
}

mixin TypedTensorMixin<T extends num> implements TypedTensor<T> {
  @override
  DeviceType get deviceType => as1d.deviceType;

  @override
  int get deviceId => as1d.deviceId;

  @override
  Device get device => as1d.device;

  @override
  int get nel => size.nel;

  @override
  T scalar([int index = 0]) => as1d[index];

  @override
  void squeeze(int dims) => size = size.squeeze(dims);

  @override
  set set(TypedTensor<T> other) {
    // TODO allow partial setting
    if (other.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    as1d.copyFrom(other.as1d);
  }

  @override
  Matrix<T> matrix(int index) {
    if (index < 0 || index >= size.numMatrices) {
      throw ArgumentError('Index out of range');
    }
    return Matrix<T>(
        OffsetTypedTensorView(this, size.numMatricesDim.unravel(index)));
  }

  // TODO accelerate this on GPU
  @override
  bool isEqual(TypedTensor<T> other, {double epsilon = 1e-8}) {
    int nel = size.nel;
    if (nel > other.size.nel) {
      nel = other.size.nel;
    }
    for (var i = 0; i < nel; i++) {
      if ((as1d[i] - other.as1d[i]).abs() > epsilon) {
        return false;
      }
    }
    return true;
  }

  // TODO accelerate this on GPU
  @override
  void assertEqual(TypedTensor<T> other, {double eps = 1e-8}) {
    int nel = size.nel;
    if (nel > other.size.nel) {
      nel = other.size.nel;
    }
    for (var i = 0; i < nel; i++) {
      final aVal = as1d[i];
      final bVal = other.as1d[i];
      final diff = (aVal - bVal).abs();
      if (diff > eps) {
        throw AssertionError(
            '@${size.unravel(i)}; $diff = $aVal - $bVal; eps: $eps');
      }
    }
  }

  @override
  Matrix<T> as2d({int colDims = 1}) => Matrix(this, colDims: colDims);

  @override
  Map<String, dynamic> toJson() => {
        'name': name,
        'size': size.toList(),
        'data': as1d.toList(),
      };

  @override
  String toString() => '$as1d';

  @override
  void printTextTable({int precision = 4}) {
    for (int i = 0; i < size.numMatrices; i++) {
      print(TableRenderer().render(matrix(i)));
    }
  }
}

mixin IntTensorMixin implements TypedTensor<int> {
  @override
  IntTensor slice(/* Dim | int | Iterable<int> */ index, {Context? context}) {
    if (index is! Dim) index = Dim.from(index);
    if (size.isIndex(index)) {
      throw ArgumentError('Index out of range');
    }

    final outSize = Dim(size.asList.skip(index.dims));
    return IntTensor(as1d.slice(index.nel * outSize.nel, outSize.nel), outSize,
        context: context);
  }

  @override
  Future<TypedTensor<int>> t({TypedTensor<int>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<TypedTensor<int>> matmul(FutureOr<TypedTensor<int>> other,
      {TypedTensor<int>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<TypedTensor<int>> matmulT(FutureOr<TypedTensor<int>> other,
      {TypedTensor<int>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<TypedTensor<int>> matmulCadd(
      FutureOr<TypedTensor<int>> other, FutureOr<TypedTensor<int>> c,
      {TypedTensor<int>? out}) {
    throw UnimplementedError();
  }

  @override
  Future<TypedTensor<int>> matmulCaddT(
      FutureOr<TypedTensor<int>> other, FutureOr<TypedTensor<int>> c,
      {TypedTensor<int>? out}) {
    throw UnimplementedError();
  }
}

class IntTensor
    with IntTensorMixin, TypedTensorMixin<int>
    implements TypedTensor<int> {
  @override
  String name = 'unnamed';

  @override
  final Onesor<int> as1d;

  Dim _size;

  IntTensor(this.as1d, this._size, {this.name = '', Context? context}) {
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

class OffsetTypedTensorView<T extends num>
    with TypedTensorMixin<T>
    implements TypedTensor<T> {
  @override
  String name = 'unnamed';

  final TypedTensor<T> _inner;

  final Dim offset;

  @override
  late final OnesorView<T> as1d =
      _inner.as1d.view(offset.nel * size.nel, size.nel);

  late Dim _size = Dim(size.asList.skip(offset.dims));

  OffsetTypedTensorView(this._inner, this.offset, {this.name = 'unnamed'}) {
    // TODO validate
    if (as1d.length != size.nel) {
      throw ArgumentError('Size does not match');
    }
  }

  @override
  Dim get size => _size;

  @override
  set size(Dim newSize) {
    if (newSize.nel != nel) {
      throw ArgumentError('Size does not match');
    }
    _size = newSize;
  }

  @override
  void release() {}

  @override
  TypedTensor<T> slice(/* Dim | int | Iterable<int> */ index,
      {Context? context}) {
    // TODO
    throw UnimplementedError();
  }

  @override
  Future<TypedTensor<T>> matmul(FutureOr<TypedTensor<T>> other,
      {TypedTensor<T>? out}) {
    // TODO: implement matmul
    throw UnimplementedError();
  }

  @override
  Future<TypedTensor<T>> matmulCadd(
      FutureOr<TypedTensor<T>> other, FutureOr<TypedTensor<T>> c,
      {TypedTensor<T>? out}) {
    // TODO: implement matmulCadd
    throw UnimplementedError();
  }

  @override
  Future<TypedTensor<T>> matmulCaddT(
      FutureOr<TypedTensor<T>> other, FutureOr<TypedTensor<T>> c,
      {TypedTensor<T>? out}) {
    // TODO: implement matmulCaddT
    throw UnimplementedError();
  }

  @override
  Future<TypedTensor<T>> matmulT(FutureOr<TypedTensor<T>> other,
      {TypedTensor<T>? out}) {
    // TODO: implement matmulT
    throw UnimplementedError();
  }

  @override
  Future<TypedTensor<T>> t({TypedTensor<T>? out}) {
    // TODO: implement t
    throw UnimplementedError();
  }
}
