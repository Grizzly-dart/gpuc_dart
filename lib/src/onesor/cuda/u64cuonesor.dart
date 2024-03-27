import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U64CuOnesor implements CuOnesor<int>, U64Onesor {
  @override
  ffi.Pointer<ffi.Uint64> get ptr;

  factory U64CuOnesor(ffi.Pointer<ffi.Uint64> ptr, int length, int deviceId,
          {Context? context}) =>
      _U64CuOnesor(ptr, length, deviceId, context: context);

  static U64CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _U64CuOnesor.sized(stream, length, context: context);

  static U64CuOnesor fromList(CudaStream stream, Uint64List list,
          {Context? context}) =>
      _U64CuOnesor.fromList(stream, list, context: context);

  static U64CuOnesor copy(CudaStream? stream, Onesor<int> other,
          {Context? context}) =>
      _U64CuOnesor.copy(other, stream: stream, context: context);

  @override
  int operator [](int index) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    return cuda.getU64(ptr, index, deviceId);
  }

  @override
  void operator []=(int index, int value) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    cuda.setU64(ptr, index, value, deviceId);
  }

  @override
  U64COnesor read({Context? context, CudaStream? stream}) {
    final ret = U64COnesor.sized(length, context: context);
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
  U64CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream(deviceId, context: lContext);
      final ret = U64CuOnesor.sized(stream, length, context: context);
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
  U64CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U64CuOnesorView(this, start, length);
  }
}

class _U64CuOnesor
    with Onesor<int>, ListMixin<int>, U64Onesor, CuOnesor<int>, U64CuOnesor
    implements U64CuOnesor {
  ffi.Pointer<ffi.Uint64> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _U64CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _U64CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr = cuda.allocate(stream, length * Uint64List.bytesPerElement);
    return _U64CuOnesor(ptr.cast(), length, stream.deviceId, context: context);
  }

  static _U64CuOnesor fromList(CudaStream stream, Uint64List list,
      {Context? context}) {
    final ret = _U64CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(U64DartOnesor(list), stream: stream);
    return ret;
  }

  static _U64CuOnesor copy(Onesor<int> other,
      {CudaStream? stream, Context? context}) {
    final lContext = Context();
    try {
      stream = stream ?? CudaStream(other.deviceId, context: lContext);
      final ret = _U64CuOnesor.sized(stream, other.length, context: context);
      ret.copyFrom(other, stream: stream);
      return ret;
    } finally {
      lContext.release();
    }
  }

  @override
  ffi.Pointer<ffi.Uint64> get ptr => _ptr;

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

class U64CuOnesorView
    with Onesor<int>, U64Onesor, ListMixin<int>, CuOnesor<int>, U64CuOnesor
    implements U64CuOnesor, CuOnesorView<int>, U64OnesorView {
  final U64CuOnesor _list;

  @override
  final int offset;

  @override
  final int length;

  U64CuOnesorView(this._list, this.offset, this.length);

  @override
  int get deviceId => _list.deviceId;

  @override
  late final ffi.Pointer<ffi.Uint64> ptr = _list.ptr + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  U64CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U64CuOnesorView(_list, start + offset, length);
  }
}
