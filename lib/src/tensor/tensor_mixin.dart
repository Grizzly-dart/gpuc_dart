import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:text_table/text_table.dart';

mixin TensorMixin implements Tensor {
  @override
  F64Ptr get ptr => as1d.ptr;

  @override
  int get nel => size.nel;

  @override
  DeviceType get deviceType => as1d.deviceType;

  @override
  int get deviceId => as1d.deviceId;

  @override
  Device get device => as1d.device;

  @override
  void squeeze(int dims) => size = size.squeeze(dims);

  @override
  double scalar([int index = 0]) => as1d[index];

  @override
  set set(Tensor other) {
    if (other.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    as1d.copyFrom(other.as1d);
  }

  // TODO start and length
  @override
  Tensor slice(/* Dim | int | Iterable<int> */ index, {Context? context}) {
    if (index is! Dim) index = Dim.from(index);
    if (size.isIndex(index)) {
      throw ArgumentError('Index out of range');
    }

    final outSize = Dim(size.asList.skip(index.dims));
    return Tensor(as1d.slice(index.nel * outSize.nel, outSize.nel), outSize,
        context: context);
  }

  @override
  Tensor operator [](dynamic /* Dim | int | Iterable<int> */ index) {
    if (index is! Dim) index = Dim.from(index);
    if (!size.isIndex(index)) {
      throw ArgumentError('Index out of range');
    }

    final outSize = Dim(size.asList.skip(index.dims));
    return TensorView(as1d.view(index.nel * outSize.nel, outSize.nel), outSize);
  }

  @override
  void operator []=(
      dynamic /* Dim | int | Iterable<int> */ index, Tensor value) {
    if (index is! Dim) index = Dim.from(index);
    if (!size.isIndex(index)) {
      throw ArgumentError('Index out of range');
    }

    final outSize = Dim(size.asList.skip(index.dims));
    if (value.size.nel != outSize.nel) {
      throw ArgumentError('Size mismatch');
    }

    as1d.view(index.nel * outSize.nel, outSize.nel).copyFrom(value.as1d);
  }

  @override
  NList row(int index, {int colDims = 1}) {
    if (size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    final size2d = size.squeeze2D(colDims: colDims);
    if (index < 0 || index >= size2d.rows) {
      throw ArgumentError('Index out of range');
    }
    return as1d.view(index * size2d.cols, size2d.cols);
  }

  @override
  Matrix as2d({int colDims = 1}) => Matrix(this, colDims: colDims);

  @override
  Matrix matrix(index) {
    if (index < 0 || index >= size.numMatrices) {
      throw ArgumentError('Index out of range');
    }
    return Matrix(TensorView(
        as1d.view(index * size.rows * size.cols, size.rows * size.cols),
        size.to2D()));
  }

  @override
  Future<Tensor> operator +(covariant FutureOr<Tensor> other) async {
    Tensor b = await other;
    if (b.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1 = CudaList.copy(as1d, stream: stream, context: ctx);
      final inp2 = CudaList.copy(b.as1d, stream: stream, context: ctx);
      final out = CudaList.sized(stream, nel, context: ctx);
      cuda.addition(
          stream, out.ptr.cast(), inp1.ptr.cast(), inp2.ptr.cast(), nel);
      final outTensor = Tensor.sized(size, name: '$name + ${b.name}');
      ctx.releaseOnErr(outTensor);
      out.copyTo(outTensor.as1d, stream: stream);
      await stream.sync();
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  @override
  Future<Tensor> operator -(covariant FutureOr<Tensor> other) async {
    Tensor b = await other;
    if (b.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1 = CudaList.copy(as1d, stream: stream, context: ctx);
      final inp2 = CudaList.copy(b.as1d, stream: stream, context: ctx);
      final out = CudaList.sized(stream, nel, context: ctx);
      cuda.subtract(
          stream, out.ptr.cast(), inp1.ptr.cast(), inp2.ptr.cast(), nel);
      final outTensor = Tensor.sized(size, name: '$name + ${b.name}');
      ctx.releaseOnErr(outTensor);
      out.copyTo(outTensor.as1d, stream: stream);
      await stream.sync();
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  @override
  Future<Tensor> operator *(covariant FutureOr<Tensor> other) async {
    Tensor b = await other;
    if (b.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1 = CudaList.copy(as1d, stream: stream, context: ctx);
      final inp2 = CudaList.copy(b.as1d, stream: stream, context: ctx);
      final out = CudaList.sized(stream, nel, context: ctx);
      cuda.multiply(
          stream, out.ptr.cast(), inp1.ptr.cast(), inp2.ptr.cast(), nel);
      final outTensor = Tensor.sized(size, name: '$name + ${b.name}');
      ctx.releaseOnErr(outTensor);
      out.copyTo(outTensor.as1d, stream: stream);
      await stream.sync();
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  @override
  Future<Tensor> operator /(covariant FutureOr<Tensor> other) async {
    Tensor b = await other;
    if (b.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1 = CudaList.copy(as1d, stream: stream, context: ctx);
      final inp2 = CudaList.copy(b.as1d, stream: stream, context: ctx);
      final out = CudaList.sized(stream, nel, context: ctx);
      cuda.divide(
          stream, out.ptr.cast(), inp1.ptr.cast(), inp2.ptr.cast(), nel);
      final outTensor = Tensor.sized(size, name: '$name + ${b.name}');
      ctx.releaseOnErr(outTensor);
      out.copyTo(outTensor.as1d, stream: stream);
      await stream.sync();
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  @override
  Tensor rearrange(List<int> order, {DeviceType? forceDeviceType}) {
    if (order.length != size.dims) {
      throw ArgumentError('Invalid order length');
    }
    final outSize = size.rearrange(order);
    // TODO detect device
    final deviceType = DeviceType.dart;
    if (deviceType == DeviceType.dart) {
      final outData = DartList.sized(outSize.nel);
      for (int i = 0; i < size.nel; i++) {
        final index = size.unravel(i);
        final outIndex = outSize.ravel(index.rearrange(order));
        outData[outIndex] = as1d[i];
      }
      final outTensor = Tensor(outData, outSize, name: 'rearrange($name)');
      return outTensor;
    } else if (deviceType == DeviceType.c) {
      // TODO
    } else if (deviceType == DeviceType.cuda) {
      /* TODO
      final outData = CList.sized(outSize.nel);
      final ctx = Context();
      try {
        int deviceId = 0; // TODO implement device selection
        final stream = CudaStream(deviceId, context: ctx);
        final inp = CudaList.copy(data, stream: stream, context: ctx);
        CudaFFI.rearrange(stream, outData.ptr.cast(), inp.ptr.cast(),
            _size.toList(), outSize.toList());
        final outTensor = Tensor(outData, outSize, name: 'rearrange($name)');
        ctx.releaseOnErr(outTensor);
        return outTensor;
      } catch (e) {
        ctx.release(isError: true);
        rethrow;
      } finally {
        ctx.release();
      }
       */
    }
    throw UnimplementedError('Device not implemented');
  }

  @override
  bool isEqual(Tensor other, {double epsilon = 1e-8}) {
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

  @override
  void assertEqual(Tensor other, {double eps = 1e-8}) {
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
  void printTextTable() {
    for (int i = 0; i < size.numMatrices; i++) {
      print(TableRenderer().render(matrix(i)));
    }
  }

  @override
  Map<String, dynamic> toJson() => {
        'name': name,
        'size': size.toList(),
        'data': as1d.toList(),
      };
}
