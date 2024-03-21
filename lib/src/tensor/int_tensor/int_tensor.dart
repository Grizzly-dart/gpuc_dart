import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:text_table/text_table.dart';

abstract mixin class Tensor<T extends num> implements Resource {
  String get name;

  set name(String name);

  Onesor<T> get as1d;

  Dim get size;

  set size(Dim size);

  int get nel => size.nel;

  DeviceType get deviceType => as1d.deviceType;

  int get deviceId => as1d.deviceId;

  Device get device => as1d.device;

  T scalar([int index = 0]) => as1d[index];

  void squeeze(int dims) => size = size.squeeze(dims);

  set set(Tensor<T> other) {
    // TODO allow partial setting
    if (other.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    as1d.copyFrom(other.as1d);
  }

  Tensor<T> slice(/* Dim | int | Iterable<int> */ index, {Context? context});

  Matrix<T> as2d({int colDims = 1}) => Matrix(this, colDims: colDims);

  Matrix<T> matrix(int index) {
    if (index < 0 || index >= size.numMatrices) {
      throw ArgumentError('Index out of range');
    }
    return Matrix<T>(
        OffsetTypedTensorView(this, size.numMatricesDim.unravel(index)));
  }

  Future<Tensor<T>> t({Tensor<T>? out});

  Future<Tensor<T>> matmul(FutureOr<Tensor<T>> other, {Tensor<T>? out});

  Future<Tensor<T>> matmulT(FutureOr<Tensor<T>> other, {Tensor<T>? out});

  Future<Tensor<T>> matmulCadd(FutureOr<Tensor<T>> other, FutureOr<Tensor<T>> c,
      {Tensor<T>? out});

  Future<Tensor<T>> matmulCaddT(
      FutureOr<Tensor<T>> other, FutureOr<Tensor<T>> c,
      {Tensor<T>? out});

  Map<String, dynamic> toJson() => {
        'name': name,
        'size': size.toList(),
        'data': as1d.toList(),
      };

  Future<Tensor<T>> pickRows(FutureOr<Tensor<int>> indices,
      {Tensor<T>? out}) async {
    final b = await indices;
    final ctx = Context();
    // TODO check if input is small enough to be
    try {
      if (cuda.exists()) {
        int deviceId = 0; // select device
        final outSize = Dim([...b.size.asList, size.cols]);
        final stream = CudaStream(deviceId, context: ctx);

        final inpBuf = CuOnesor.copy(as1d, stream: stream, context: ctx);
        final indicesBuf = CuOnesor.copy(b.as1d, stream: stream, context: ctx);
        final outBuf = CuOnesor.sized(stream, outSize.nel, context: ctx);
        cuda.pickRows(stream, outBuf.ptr, inpBuf.ptr, indicesBuf.ptr,
            outSize.squeeze2D());
        out ??= Tensor<T>(outBuf, outSize, context: ctx);
        return out;
      }
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  // TODO accelerate this on GPU
  bool isEqual(Tensor<T> other, {double epsilon = 1e-8}) {
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
  void assertEqual(Tensor<T> other, {double eps = 1e-8}) {
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
  String toString() => '$as1d';

  /* TODO , int? tableWidth, int? maxChars*/
  void printTextTable({int precision = 4}) {
    for (int i = 0; i < size.numMatrices; i++) {
      print(TableRenderer().render(matrix(i)));
    }
  }
}

mixin IntTensorMixin implements Tensor<int> {
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

class IntTensor with Tensor<int>, IntTensorMixin implements Tensor<int> {
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

class OffsetTypedTensorView<T extends num> with Tensor<T> implements Tensor<T> {
  @override
  String name = 'unnamed';

  final Tensor<T> _inner;

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
  Tensor<T> slice(/* Dim | int | Iterable<int> */ index, {Context? context}) {
    // TODO
    throw UnimplementedError();
  }

  @override
  Future<Tensor<T>> matmul(FutureOr<Tensor<T>> other, {Tensor<T>? out}) {
    // TODO: implement matmul
    throw UnimplementedError();
  }

  @override
  Future<Tensor<T>> matmulCadd(FutureOr<Tensor<T>> other, FutureOr<Tensor<T>> c,
      {Tensor<T>? out}) {
    // TODO: implement matmulCadd
    throw UnimplementedError();
  }

  @override
  Future<Tensor<T>> matmulCaddT(
      FutureOr<Tensor<T>> other, FutureOr<Tensor<T>> c,
      {Tensor<T>? out}) {
    // TODO: implement matmulCaddT
    throw UnimplementedError();
  }

  @override
  Future<Tensor<T>> matmulT(FutureOr<Tensor<T>> other, {Tensor<T>? out}) {
    // TODO: implement matmulT
    throw UnimplementedError();
  }

  @override
  Future<Tensor<T>> t({Tensor<T>? out}) {
    // TODO: implement t
    throw UnimplementedError();
  }
}
