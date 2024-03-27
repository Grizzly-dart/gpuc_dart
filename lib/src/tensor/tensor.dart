import 'dart:async';

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:text_table/text_table.dart';

export 'dim.dart';
export 'matrix.dart';
export 'tensor_future.dart';
export 'tensor2d_mixin.dart';
export 'tensor_view.dart';
export 'f64/f64tensor.dart';
export 'f32/f32tensor.dart';
export 'i64/i64tensor.dart';
export 'u64/u64tensor.dart';
export 'i32/i32tensor.dart';
export 'u32/u32tensor.dart';
export 'i16/i16tensor.dart';
export 'u16/u16tensor.dart';
export 'i8/i8tensor.dart';
export 'u8/u8tensor.dart';

abstract mixin class Tensor<T extends num> implements Resource {
  factory Tensor(Onesor<T> as1d, Dim size, {String name = ''}) {
    switch (as1d.type) {
      case f64:
        return F64Tensor(as1d as F64Onesor, size, name: name) as Tensor<T>;
      case f32:
        return F32Tensor(as1d as F32Onesor, size, name: name) as Tensor<T>;
      case i64:
        return I64Tensor(as1d as I64Onesor, size, name: name) as Tensor<T>;
      case u64:
        return U64Tensor(as1d as U64Onesor, size, name: name) as Tensor<T>;
      case i32:
        return I32Tensor(as1d as I32Onesor, size, name: name) as Tensor<T>;
      case u32:
        return U32Tensor(as1d as U32Onesor, size, name: name) as Tensor<T>;
      case i16:
        return I16Tensor(as1d as I16Onesor, size, name: name) as Tensor<T>;
      case u16:
        return U16Tensor(as1d as U16Onesor, size, name: name) as Tensor<T>;
      case i8:
        return I8Tensor(as1d as I8Onesor, size, name: name) as Tensor<T>;
      case u8:
        return U8Tensor(as1d as U8Onesor, size, name: name) as Tensor<T>;
      default:
        throw ArgumentError('Unsupported type: ${as1d.type}');
    }
  }

  factory Tensor.sized(Dim size, NumType<T> type, {String name = ''}) {
    switch (type) {
      case f64:
        return F64Tensor.sized(size, name: name) as Tensor<T>;
      case f32:
        return F32Tensor.sized(size, name: name) as Tensor<T>;
      case i64:
        return I64Tensor.sized(size, name: name) as Tensor<T>;
      case u64:
        return U64Tensor.sized(size, name: name) as Tensor<T>;
      case i32:
        return I32Tensor.sized(size, name: name) as Tensor<T>;
      case u32:
        return U32Tensor.sized(size, name: name) as Tensor<T>;
      case i16:
        return I16Tensor.sized(size, name: name) as Tensor<T>;
      case u16:
        return U16Tensor.sized(size, name: name) as Tensor<T>;
      case i8:
        return I8Tensor.sized(size, name: name) as Tensor<T>;
      case u8:
        return U8Tensor.sized(size, name: name) as Tensor<T>;
      default:
        throw ArgumentError('Unsupported type: $type');
    }
  }

  String get name;

  set name(String name);

  Onesor<T> get as1d;

  NumType<T> get type => as1d.type;

  Dim get size;

  set size(Dim size);

  int get nel => size.nel;

  int get lengthBytes => as1d.lengthBytes;

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

  TensorView<T> operator [](dynamic /* Dim | int | Iterable<int> */ index) {
    throw UnimplementedError();
  }

  void operator []=(
      dynamic /* Dim | int | Iterable<int> */ index, Tensor<T> value) {
    if (index is! Dim) index = Dim.from(index);
    if (!size.isIndex(index)) {
      throw ArgumentError('Index out of range');
    }

    final outSize = Dim(size.asList.skip(index.dims));
    if (value.size.dims == outSize.dims) {
      if (value.size != outSize) {
        throw ArgumentError('Size mismatch');
      }
    } else if (value.size.dims == outSize.dims + 1) {
      if (!value.size.asList.skip(1).isEqual(outSize.asList)) {
        throw ArgumentError('Size mismatch');
      }
      if (value.size.asList[0] + index.asList.last >
          size.asList[index.dims - 1]) {
        throw ArgumentError('Size mismatch');
      }
    } else {
      throw ArgumentError('Size mismatch');
    }
    as1d
        .view((index.asList * size.strides).sum, value.nel)
        .copyFrom(value.as1d);
  }

  Matrix<T> as2d({int colDims = 1}) => Matrix(this, colDims: colDims);

  Matrix<T> matrix(int index) {
    if (index < 0 || index >= size.numMatrices) {
      throw ArgumentError('Index out of range');
    }
    final matIndex = size.numMatricesDim.unravel(index);
    return Matrix<T>(this[matIndex]);
  }

  Onesor<T> row(int index, {int colDims = 1}) {
    if (size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    final size2d = size.squeeze2D(colDims: colDims);
    if (index < 0 || index >= size2d.rows) {
      throw ArgumentError('Index out of range');
    }
    return as1d.view(index * size2d.cols, size2d.cols);
  }

  Future<Tensor> plus_(FutureOr<Tensor> other) => plus(other, out: this);

  Future<Tensor> plus(FutureOr<Tensor> other, {Tensor? out}) async {
    final b = await other;
    if (b.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    if (out != null && out.nel != nel) {
      throw ArgumentError('Output size mismatch');
    }
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1Buf = CuOnesor.copy(stream, as1d, context: ctx);
      final inp2Buf = CuOnesor.copy(stream, b.as1d, context: ctx);
      final outType = type.bytes > b.type.bytes ? type : b.type;
      final outBuf = CuOnesor.sized(stream, outType, nel, context: ctx);
      cuda.addition(stream, outBuf, inp1Buf, inp2Buf, nel);
      if (out == null) {
        out = Tensor.sized(size, outType, name: '$name + ${b.name}');
        ctx.releaseOnErr(out);
      }
      outBuf.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor> operator +(FutureOr<Tensor> other) async {
    final b = await other;
    if (b.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1Buf = CuOnesor.copy(stream, as1d, context: ctx);
      final inp2Buf = CuOnesor.copy(stream, b.as1d, context: ctx);
      final outType = type.bytes > b.type.bytes ? type : b.type;
      final outBuf = CuOnesor.sized(stream, outType, nel, context: ctx);
      cuda.addition(stream, outBuf, inp1Buf, inp2Buf, nel);
      final out = outBuf.read(stream: stream);
      ctx.releaseOnErr(out);
      final outTensor = out.toTensor(size, name: '$name + ${b.name}');
      await stream.sync();
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor> operator -(FutureOr<Tensor> other) async {
    final b = await other;
    if (b.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1Buf = CuOnesor.copy(stream, as1d, context: ctx);
      final inp2Buf = CuOnesor.copy(stream, b.as1d, context: ctx);
      final outType = type.bytes > b.type.bytes ? type : b.type;
      final outBuf = CuOnesor.sized(stream, outType, nel, context: ctx);
      cuda.sub(stream, outBuf, inp1Buf, inp2Buf, nel);
      final out = outBuf.read(stream: stream);
      ctx.releaseOnErr(out);
      final outTensor = out.toTensor(size, name: '$name - ${b.name}');
      await stream.sync();
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor> operator *(FutureOr<Tensor> other) async {
    final b = await other;
    if (b.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1Buf = CuOnesor.copy(stream, as1d, context: ctx);
      final inp2Buf = CuOnesor.copy(stream, b.as1d, context: ctx);
      final outType = type.bytes > b.type.bytes ? type : b.type;
      final outBuf = CuOnesor.sized(stream, outType, nel, context: ctx);
      cuda.mul(stream, outBuf, inp1Buf, inp2Buf, nel);
      final out = outBuf.read(stream: stream);
      ctx.releaseOnErr(out);
      final outTensor = out.toTensor(size, name: '$name * ${b.name}');
      await stream.sync();
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor> operator /(FutureOr<Tensor> other) async {
    final b = await other;
    if (b.nel != nel) {
      throw ArgumentError('Size mismatch');
    }
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1Buf = CuOnesor.copy(stream, as1d, context: ctx);
      final inp2Buf = CuOnesor.copy(stream, b.as1d, context: ctx);
      final outType = type.bytes > b.type.bytes ? type : b.type;
      final outBuf = CuOnesor.sized(stream, outType, nel, context: ctx);
      cuda.div(stream, outBuf, inp1Buf, inp2Buf, nel);
      final out = outBuf.read(stream: stream);
      ctx.releaseOnErr(out);
      final outTensor = out.toTensor(size, name: '$name / ${b.name}');
      await stream.sync();
      return outTensor;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> sin({Tensor<double>? out}) async {
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1Buf = CuOnesor.copy(stream, as1d, context: ctx);
      final outBuf =
          CuOnesor.sized(stream, out?.type ?? f64, nel, context: ctx);
      cuda.sin(stream, outBuf, inp1Buf, nel);
      if (out == null) {
        out = F64Tensor.sized(size, name: 'sin($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != nel) {
          throw ArgumentError('Output size mismatch');
        }
      }
      outBuf.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> cos({Tensor<double>? out}) async {
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1Buf = CuOnesor.copy(stream, as1d, context: ctx);
      final outBuf =
          CuOnesor.sized(stream, out?.type ?? f64, nel, context: ctx);
      cuda.cos(stream, outBuf, inp1Buf, nel);
      if (out == null) {
        out = F64Tensor.sized(size, name: 'sin($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != nel) {
          throw ArgumentError('Output size mismatch');
        }
      }
      outBuf.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> tan({Tensor<double>? out}) async {
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1Buf = CuOnesor.copy(stream, as1d, context: ctx);
      final outBuf =
          CuOnesor.sized(stream, out?.type ?? f64, nel, context: ctx);
      cuda.tan(stream, outBuf, inp1Buf, nel);
      if (out == null) {
        out = F64Tensor.sized(size, name: 'sin($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != nel) {
          throw ArgumentError('Output size mismatch');
        }
      }
      outBuf.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> sinh({Tensor<double>? out}) async {
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1Buf = CuOnesor.copy(stream, as1d, context: ctx);
      final outBuf =
          CuOnesor.sized(stream, out?.type ?? f64, nel, context: ctx);
      cuda.sinh(stream, outBuf, inp1Buf, nel);
      if (out == null) {
        out = F64Tensor.sized(size, name: 'sin($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != nel) {
          throw ArgumentError('Output size mismatch');
        }
      }
      outBuf.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> cosh({Tensor<double>? out}) async {
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1Buf = CuOnesor.copy(stream, as1d, context: ctx);
      final outBuf =
          CuOnesor.sized(stream, out?.type ?? f64, nel, context: ctx);
      cuda.cosh(stream, outBuf, inp1Buf, nel);
      if (out == null) {
        out = F64Tensor.sized(size, name: 'sin($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != nel) {
          throw ArgumentError('Output size mismatch');
        }
      }
      outBuf.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> tanh({Tensor<double>? out}) async {
    final ctx = Context();
    try {
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      // TODO implement split processing if not all data fits into memory or to maximize parallelism
      final inp1Buf = CuOnesor.copy(stream, as1d, context: ctx);
      final outBuf =
          CuOnesor.sized(stream, out?.type ?? f64, nel, context: ctx);
      cuda.tanh(stream, outBuf, inp1Buf, nel);
      if (out == null) {
        out = F64Tensor.sized(size, name: 'sin($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != nel) {
          throw ArgumentError('Output size mismatch');
        }
      }
      outBuf.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> sumRows({int colDims = 1, Tensor<double>? out}) async {
    if (size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    Dim inpSize = size.squeeze2D(colDims: colDims);
    Dim outSize = Dim2(inpSize.rows, 1);
    final ctx = Context();
    try {
      // TODO implement Dart summing for web
      // TODO implement C summing for non-web
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      final inp = CuOnesor.copy(stream, as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, outSize.nel, context: ctx);
      cuda.sum2d(stream, outCuda, inp, inpSize.to2D());
      if (out == null) {
        out = F64Tensor.sized(outSize, name: 'sum2D($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != outSize.nel) {
          throw ArgumentError('Output size mismatch');
        }
      }
      outCuda.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> meanRows(
      {int colDims = 1, Tensor<double>? out}) async {
    if (size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    Dim inpSize = size.squeeze2D(colDims: colDims);
    Dim outSize = Dim2(inpSize.rows, 1);
    final ctx = Context();
    try {
      // TODO implement Dart summing for web
      // TODO implement C summing for non-web
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      final inp = CuOnesor.copy(stream, as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, outSize.nel, context: ctx);
      cuda.mean2d(stream, outCuda, inp, inpSize.to2D());
      if (out == null) {
        out = F64Tensor.sized(outSize, name: 'sum2D($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != outSize.nel) {
          throw ArgumentError('Output size mismatch');
        }
      }
      outCuda.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> varianceRows(
      {int colDims = 1, Tensor<double>? out, int correction = 0}) async {
    if (size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    Dim inpSize = size.squeeze2D(colDims: colDims);
    Dim outSize = Dim2(inpSize.rows, 1);
    final ctx = Context();
    try {
      // TODO implement Dart summing for web
      // TODO implement C summing for non-web
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      final inp = CuOnesor.copy(stream, as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, outSize.nel, context: ctx);
      cuda.variance2d(stream, outCuda, inp, inpSize.to2D(), correction);
      if (out == null) {
        out = F64Tensor.sized(outSize, name: 'sum2D($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != outSize.nel) {
          throw ArgumentError('Output size mismatch');
        }
      }
      outCuda.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> stdRows(
      {int colDims = 1, Tensor<double>? out, int correction = 0}) async {
    if (size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    Dim inpSize = size.squeeze2D(colDims: colDims);
    Dim outSize = Dim2(inpSize.rows, 1);
    final ctx = Context();
    try {
      // TODO implement Dart summing for web
      // TODO implement C summing for non-web
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      final inp = CuOnesor.copy(stream, as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, outSize.nel, context: ctx);
      cuda.std2d(stream, outCuda, inp, inpSize.to2D(), correction);
      if (out == null) {
        out = F64Tensor.sized(outSize, name: 'sum2D($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != outSize.nel) {
          throw ArgumentError('Output size mismatch');
        }
      }
      outCuda.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<double>> normalizeRows(
      {int colDims = 1, Tensor<double>? out, double epsilon = 1e-5}) async {
    if (size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    Dim inpSize = size.squeeze2D(colDims: colDims);
    Dim outSize = Dim2(inpSize.rows, 1);
    final ctx = Context();
    try {
      // TODO implement Dart summing for web
      // TODO implement C summing for non-web
      int deviceId = 0; // TODO implement device selection
      final stream = CudaStream(deviceId, context: ctx);
      final inp = CuOnesor.copy(stream, as1d, context: ctx);
      final outCuda = F64CuOnesor.sized(stream, outSize.nel, context: ctx);
      cuda.normalize2d(stream, outCuda, inp, inpSize.to2D(), epsilon);
      if (out == null) {
        out = F64Tensor.sized(outSize, name: 'sum2D($name)');
        ctx.releaseOnErr(out);
      } else {
        if (out.nel != outSize.nel) {
          throw ArgumentError('Output size mismatch');
        }
      }
      outCuda.copyTo(out.as1d, stream: stream);
      await stream.sync();
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }

  Future<Tensor<T>> t({Tensor<T>? out});

  Future<Tensor<T>> mm(FutureOr<Tensor<T>> other, {Tensor<T>? out});

  Future<Tensor<T>> mmAt(FutureOr<Tensor<T>> other, {Tensor<T>? out}) {
    throw UnimplementedError();
  }

  Future<Tensor<T>> mmBt(FutureOr<Tensor<T>> other, {Tensor<T>? out});

  Future<Tensor<T>> mmColAdd(FutureOr<Tensor<T>> other, FutureOr<Tensor<T>> c,
      {Tensor<T>? out});

  Future<Tensor<T>> mmBtColAdd(FutureOr<Tensor<T>> other, FutureOr<Tensor<T>> c,
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

        final inpBuf = CuOnesor.copy(stream, as1d, context: ctx);
        final indicesBuf = CuOnesor.copy(stream, b.as1d, context: ctx);
        final outBuf = CuOnesor.sized(stream, type, outSize.nel, context: ctx);
        cuda.pickRows(stream, outBuf.ptr, inpBuf.ptr, indicesBuf.ptr,
            outSize.squeeze2D());
        if (out != null) {
          outBuf.copyTo(out.as1d, stream: stream);
        } else {
          final outOnesor = outBuf.read(stream: stream);
          ctx.releaseOnErr(outOnesor);
          out = outOnesor.toTensor(outSize);
        }
        await stream.sync();
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

/*
  @override
  void rearrange(List<int> order, {DeviceType? forceDeviceType}) {
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
   */
}
