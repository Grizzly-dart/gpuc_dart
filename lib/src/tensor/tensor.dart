import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/onesor/onesor.dart';
import 'package:text_table/text_table.dart';

export 'dim.dart';
export 'matrix.dart';
export 'tensor_future.dart';
export 'tensor2d_mixin.dart';
export 'tensor_view.dart';
export 'f64/f64tensor.dart';
export 'u64/u64tensor.dart';

abstract mixin class Tensor<T extends num> implements Resource {
  static Tensor<T> ccopy<T extends num>(Onesor<T> other, ) {
    // TODO
  }

  String get name;

  set name(String name);

  Onesor<T> get as1d;

  NumType<T> get numType => as1d.numType;

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
        final outBuf =
            CuOnesor.sized(stream, numType, outSize.nel, context: ctx);
        cuda.pickRows(stream, outBuf.ptr, inpBuf.ptr, indicesBuf.ptr,
            outSize.squeeze2D());
        if(out != null) {
          outBuf.copyTo(out.as1d, stream: stream);
        } else {
          final outOnesor = outBuf.read(stream: stream);
          ctx.releaseOnErr(outOnesor);
          out = outOnesor.toTensor(outSize);
        }
        return out;
      }

      throw UnimplementedError(
          'pickRows on CPU(C/Dart is not implemented yet!');
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
