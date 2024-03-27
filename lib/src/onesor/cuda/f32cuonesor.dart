import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class F32CuOnesor implements CuOnesor<double>, F32Onesor {
  @override
  ffi.Pointer<ffi.Float> get ptr;

  factory F32CuOnesor(ffi.Pointer<ffi.Float> ptr, int length, int deviceId,
          {Context? context}) =>
      _F32CuOnesor(ptr, length, deviceId, context: context);

  static F32CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _F32CuOnesor.sized(stream, length, context: context);

  static F32CuOnesor fromList(CudaStream stream, Float32List list,
          {Context? context}) =>
      _F32CuOnesor.fromList(stream, list, context: context);

  static F32CuOnesor copy(CudaStream? stream, Onesor<double> other,
          {Context? context}) =>
      _F32CuOnesor.copy(other, stream: stream, context: context);

  @override
  double operator [](int index) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    return cuda.getF32(ptr, index, deviceId);
  }

  @override
  void operator []=(int index, double value) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    cuda.setF32(ptr, index, value, deviceId);
  }

  @override
  COnesor<double> read({Context? context, CudaStream? stream}) {
    final ret = F32COnesor.sized(length, context: context);
    final lContext = Context();
    try {
      stream = stream ?? CudaStream(deviceId, context: lContext);
      cuda.memcpy(stream, ret.ptr.cast(), ptr.cast(), ret.lengthBytes);
      return ret;
    } finally {
      lContext.release();
    }
  }

  @override
  F32CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream(deviceId, context: lContext);
      final ret = F32CuOnesor.sized(stream, length, context: context);
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
  F32CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return F32CuOnesorView(this, start, length);
  }
}

class _F32CuOnesor
    with
        Onesor<double>,
        ListMixin<double>,
        F32Onesor,
        CuOnesor<double>,
        F32CuOnesor
    implements F32CuOnesor {
  ffi.Pointer<ffi.Float> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _F32CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _F32CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr = cuda.allocate(stream, length * Float32List.bytesPerElement);
    return _F32CuOnesor(ptr.cast(), length, stream.deviceId, context: context);
  }

  static _F32CuOnesor fromList(CudaStream stream, Float32List list,
      {Context? context}) {
    final ret = _F32CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(F32DartOnesor(list), stream: stream);
    return ret;
  }

  static _F32CuOnesor copy(Onesor<double> other,
      {CudaStream? stream, Context? context}) {
    final lContext = Context();
    try {
      stream = stream ?? CudaStream(other.deviceId, context: lContext);
      final ret = _F32CuOnesor.sized(stream, other.length, context: context);
      ret.copyFrom(other, stream: stream);
      return ret;
    } finally {
      lContext.release();
    }
  }

  @override
  ffi.Pointer<ffi.Float> get ptr => _ptr;

  @override
  void release({CudaStream? stream}) {
    if (_ptr == ffi.nullptr) return;
    final ctx = Context();
    try {
      stream ??= CudaStream(deviceId, context: ctx);
      cuda.memFree(stream, _ptr.cast());
      _ptr = ffi.nullptr;
    } finally {
      ctx.release();
    }
  }
}

class F32CuOnesorView
    with
        Onesor<double>,
        F32Onesor,
        ListMixin<double>,
        CuOnesor<double>,
        F32CuOnesor
    implements F32CuOnesor, CuOnesorView<double>, F32OnesorView {
  final CuOnesor<double> _list;

  @override
  final int offset;

  @override
  final int length;

  F32CuOnesorView(this._list, this.offset, this.length);

  @override
  int get deviceId => _list.deviceId;

  @override
  late final ffi.Pointer<ffi.Float> ptr =
      _list.ptr.cast<ffi.Float>() + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  F32CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return F32CuOnesorView(_list, start + offset, length);
  }
}
