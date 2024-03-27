import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I16CuOnesor implements CuOnesor<int>, I16Onesor {
  @override
  ffi.Pointer<ffi.Int16> get ptr;

  factory I16CuOnesor(ffi.Pointer<ffi.Int16> ptr, int length, int deviceId,
      {Context? context}) =>
      _I16CuOnesor(ptr, length, deviceId, context: context);

  static I16CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _I16CuOnesor.sized(stream, length, context: context);

  static I16CuOnesor fromList(CudaStream stream, Int16List list,
      {Context? context}) =>
      _I16CuOnesor.fromList(stream, list, context: context);

  static I16CuOnesor copy(CudaStream? stream, Onesor<int> other,
      {Context? context}) =>
      _I16CuOnesor.copy(other, stream: stream, context: context);

  @override
  int operator [](int index) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    return cuda.getI16(ptr, index, deviceId);
  }

  @override
  void operator []=(int index, int value) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    cuda.setI16(ptr, index, value, deviceId);
  }

  @override
  I16COnesor read({Context? context, CudaStream? stream}) {
    final ret = I16COnesor.sized(length, context: context);
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
  I16CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream(deviceId, context: lContext);
      final ret = I16CuOnesor.sized(stream, length, context: context);
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
  I16CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I16CuOnesorView(this, start, length);
  }
}

class _I16CuOnesor
    with
        Onesor<int>,
        ListMixin<int>,
        I16Onesor,
        CuOnesor<int>,
        I16CuOnesor
    implements I16CuOnesor {
  ffi.Pointer<ffi.Int16> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _I16CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _I16CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr = cuda.allocate(stream, length * Int16List.bytesPerElement);
    return _I16CuOnesor(ptr.cast(), length, stream.deviceId, context: context);
  }

  static _I16CuOnesor fromList(CudaStream stream, Int16List list,
      {Context? context}) {
    final ret = _I16CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(I16DartOnesor(list), stream: stream);
    return ret;
  }

  static _I16CuOnesor copy(Onesor<int> other,
      {CudaStream? stream, Context? context}) {
    final lContext = Context();
    try {
      stream = stream ?? CudaStream(other.deviceId, context: lContext);
      final ret = _I16CuOnesor.sized(stream, other.length, context: context);
      ret.copyFrom(other, stream: stream);
      return ret;
    } finally {
      lContext.release();
    }
  }

  @override
  ffi.Pointer<ffi.Int16> get ptr => _ptr;

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

class I16CuOnesorView
    with
        Onesor<int>,
        I16Onesor,
        ListMixin<int>,
        CuOnesor<int>,
        I16CuOnesor
    implements I16CuOnesor, CuOnesorView<int>, I16OnesorView {
  final I16CuOnesor _list;

  @override
  final int offset;

  @override
  final int length;

  I16CuOnesorView(this._list, this.offset, this.length);

  @override
  int get deviceId => _list.deviceId;

  @override
  late final ffi.Pointer<ffi.Int16> ptr =
      _list.ptr.cast<ffi.Int16>() + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  I16CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I16CuOnesorView(_list, start + offset, length);
  }
}
