import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class F64CuOnesor implements CuOnesor<double>, F64Onesor {
  @override
  ffi.Pointer<ffi.Double> get ptr;

  factory F64CuOnesor(ffi.Pointer<ffi.Double> ptr, int length, int deviceId,
          {Context? context}) =>
      _F64CuOnesor(ptr, length, deviceId, context: context);

  static F64CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _F64CuOnesor.sized(stream, length, context: context);

  static F64CuOnesor fromList(CudaStream stream, Float64List list,
          {Context? context}) =>
      _F64CuOnesor.fromList(stream, list, context: context);

  static F64CuOnesor copy(CudaStream? stream, Onesor<double> other,
          {Context? context}) =>
      _F64CuOnesor.copy(other, stream: stream, context: context);

  @override
  double operator [](int index) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    return cuda.getOne(CudaStream.noStream(deviceId), ptr, type, index: index);
  }

  @override
  void operator []=(int index, double value) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    cuda.setOne(CudaStream.noStream(deviceId), ptr, value, type, index: index);
  }

  @override
  COnesor<double> read({Context? context, CudaStream? stream}) {
    final ret = F64COnesor.sized(length, context: context);
      stream = stream ?? CudaStream.noStream(deviceId);
      cuda.memcpy(stream, ret.ptr.cast(), ptr.cast(), ret.lengthBytes);
      return ret;
  }

  @override
  F64CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream.noStream(deviceId, context: lContext);
      final ret = F64CuOnesor.sized(stream, length, context: context);
      lContext.releaseOnErr(ret);
      cuda.memcpy(stream, ret.ptr.cast(), (ptr + bytesPerItem).cast(),
          length * bytesPerItem);
      return ret;
    } catch (e) {
      lContext.release(isError: true);
      rethrow;
    } finally {
      lContext.release();
    }
  }

  @override
  F64CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return F64CuOnesorView(this, start, length);
  }
}

class _F64CuOnesor
    with
        Onesor<double>,
        ListMixin<double>,
        F64Onesor,
        CuOnesor<double>,
        F64CuOnesor
    implements F64CuOnesor {
  ffi.Pointer<ffi.Double> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _F64CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _F64CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr = cuda.allocate(stream, length * Float64List.bytesPerElement);
    return _F64CuOnesor(ptr.cast(), length, stream.deviceId, context: context);
  }

  static _F64CuOnesor fromList(CudaStream stream, Float64List list,
      {Context? context}) {
    final ret = _F64CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(F64DartOnesor(list), stream: stream);
    return ret;
  }

  static _F64CuOnesor copy(Onesor<double> other,
      {CudaStream? stream, Context? context}) {
    stream = stream ?? CudaStream.noStream(other.deviceId);
    final ret = _F64CuOnesor.sized(stream, other.length, context: context);
    ret.copyFrom(other, stream: stream);
    return ret;
  }

  @override
  ffi.Pointer<ffi.Double> get ptr => _ptr;

  @override
  void release({CudaStream? stream}) {
    if (_ptr == ffi.nullptr) return;
    stream ??= CudaStream.noStream(deviceId);
    cuda.memFree(stream, _ptr.cast());
    _ptr = ffi.nullptr;
  }
}

class F64CuOnesorView
    with
        Onesor<double>,
        F64Onesor,
        ListMixin<double>,
        CuOnesor<double>,
        F64CuOnesor
    implements F64CuOnesor, CuOnesorView<double>, F64OnesorView {
  final CuOnesor<double> _list;

  @override
  final int offset;

  @override
  final int length;

  F64CuOnesorView(this._list, this.offset, this.length);

  @override
  int get deviceId => _list.deviceId;

  @override
  late final ffi.Pointer<ffi.Double> ptr =
      _list.ptr.cast<ffi.Double>() + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  F64CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return F64CuOnesorView(_list, start + offset, length);
  }
}
