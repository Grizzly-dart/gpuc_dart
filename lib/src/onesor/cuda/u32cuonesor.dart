import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U32CuOnesor implements CuOnesor<int>, U32Onesor {
  @override
  ffi.Pointer<ffi.Uint32> get ptr;

  factory U32CuOnesor(ffi.Pointer<ffi.Uint32> ptr, int length, int deviceId,
          {Context? context}) =>
      _U32CuOnesor(ptr, length, deviceId, context: context);

  static U32CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _U32CuOnesor.sized(stream, length, context: context);

  static U32CuOnesor fromList(CudaStream stream, Uint32List list,
          {Context? context}) =>
      _U32CuOnesor.fromList(stream, list, context: context);

  static U32CuOnesor copy(CudaStream? stream, Onesor<int> other,
          {Context? context}) =>
      _U32CuOnesor.copy(other, stream: stream, context: context);

  @override
  int operator [](int index) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    return cuda.getU32(ptr, index, deviceId);
  }

  @override
  void operator []=(int index, int value) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    cuda.setU32(ptr, index, value, deviceId);
  }

  @override
  U32COnesor read({Context? context, CudaStream? stream}) {
    final ret = U32COnesor.sized(length, context: context);
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
  U32CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream(deviceId, context: lContext);
      final ret = U32CuOnesor.sized(stream, length, context: context);
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
  U32CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U32CuOnesorView(this, start, length);
  }
}

class _U32CuOnesor
    with Onesor<int>, ListMixin<int>, U32Onesor, CuOnesor<int>, U32CuOnesor
    implements U32CuOnesor {
  ffi.Pointer<ffi.Uint32> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _U32CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _U32CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr = cuda.allocate(stream, length * Uint32List.bytesPerElement);
    return _U32CuOnesor(ptr.cast(), length, stream.deviceId, context: context);
  }

  static _U32CuOnesor fromList(CudaStream stream, Uint32List list,
      {Context? context}) {
    final ret = _U32CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(U32DartOnesor(list), stream: stream);
    return ret;
  }

  static _U32CuOnesor copy(Onesor<int> other,
      {CudaStream? stream, Context? context}) {
    final lContext = Context();
    try {
      stream = stream ?? CudaStream(other.deviceId, context: lContext);
      final ret = _U32CuOnesor.sized(stream, other.length, context: context);
      ret.copyFrom(other, stream: stream);
      return ret;
    } finally {
      lContext.release();
    }
  }

  @override
  ffi.Pointer<ffi.Uint32> get ptr => _ptr;

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

class U32CuOnesorView
    with Onesor<int>, U32Onesor, ListMixin<int>, CuOnesor<int>, U32CuOnesor
    implements U32CuOnesor, CuOnesorView<int>, U32OnesorView {
  final U32CuOnesor _list;

  @override
  final int offset;

  @override
  final int length;

  U32CuOnesorView(this._list, this.offset, this.length);

  @override
  int get deviceId => _list.deviceId;

  @override
  late final ffi.Pointer<ffi.Uint32> ptr = _list.ptr + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  U32CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U32CuOnesorView(_list, start + offset, length);
  }
}
