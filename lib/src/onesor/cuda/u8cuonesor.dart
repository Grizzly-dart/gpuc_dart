import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U8CuOnesor implements CuOnesor<int>, U8Onesor {
  @override
  ffi.Pointer<ffi.Uint8> get ptr;

  factory U8CuOnesor(ffi.Pointer<ffi.Uint8> ptr, int length, int deviceId,
          {Context? context}) =>
      _U8CuOnesor(ptr, length, deviceId, context: context);

  static U8CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _U8CuOnesor.sized(stream, length, context: context);

  static U8CuOnesor fromList(CudaStream stream, Uint8List list,
          {Context? context}) =>
      _U8CuOnesor.fromList(stream, list, context: context);

  static U8CuOnesor copy(CudaStream? stream, Onesor<int> other,
          {Context? context}) =>
      _U8CuOnesor.copy(other, stream: stream, context: context);

  @override
  int operator [](int index) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    return cuda.getU8(ptr, index, deviceId);
  }

  @override
  void operator []=(int index, int value) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    cuda.setU8(ptr, index, value, deviceId);
  }

  @override
  U8COnesor read({Context? context, CudaStream? stream}) {
    final ret = U8COnesor.sized(length, context: context);
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
  U8CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream(deviceId, context: lContext);
      final ret = U8CuOnesor.sized(stream, length, context: context);
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
  U8CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U8CuOnesorView(this, start, length);
  }
}

class _U8CuOnesor
    with Onesor<int>, ListMixin<int>, U8Onesor, CuOnesor<int>, U8CuOnesor
    implements U8CuOnesor {
  ffi.Pointer<ffi.Uint8> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _U8CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _U8CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr = cuda.allocate(stream, length * Uint8List.bytesPerElement);
    return _U8CuOnesor(ptr.cast(), length, stream.deviceId, context: context);
  }

  static _U8CuOnesor fromList(CudaStream stream, Uint8List list,
      {Context? context}) {
    final ret = _U8CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(U8DartOnesor(list), stream: stream);
    return ret;
  }

  static _U8CuOnesor copy(Onesor<int> other,
      {CudaStream? stream, Context? context}) {
    final lContext = Context();
    try {
      stream = stream ?? CudaStream(other.deviceId, context: lContext);
      final ret = _U8CuOnesor.sized(stream, other.length, context: context);
      ret.copyFrom(other, stream: stream);
      return ret;
    } finally {
      lContext.release();
    }
  }

  @override
  ffi.Pointer<ffi.Uint8> get ptr => _ptr;

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

class U8CuOnesorView
    with Onesor<int>, U8Onesor, ListMixin<int>, CuOnesor<int>, U8CuOnesor
    implements U8CuOnesor, CuOnesorView<int>, U8OnesorView {
  final U8CuOnesor _list;

  @override
  final int offset;

  @override
  final int length;

  U8CuOnesorView(this._list, this.offset, this.length);

  @override
  int get deviceId => _list.deviceId;

  @override
  late final ffi.Pointer<ffi.Uint8> ptr = _list.ptr + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  U8CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U8CuOnesorView(_list, start + offset, length);
  }
}
