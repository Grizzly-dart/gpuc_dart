import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I8CuOnesor implements CuOnesor<int>, I8Onesor {
  @override
  ffi.Pointer<ffi.Int8> get ptr;

  factory I8CuOnesor(ffi.Pointer<ffi.Int8> ptr, int length, int deviceId,
          {Context? context}) =>
      _I8CuOnesor(ptr, length, deviceId, context: context);

  static I8CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _I8CuOnesor.sized(stream, length, context: context);

  static I8CuOnesor fromList(CudaStream stream, Int8List list,
          {Context? context}) =>
      _I8CuOnesor.fromList(stream, list, context: context);

  static I8CuOnesor copy(CudaStream? stream, Onesor<int> other,
          {Context? context}) =>
      _I8CuOnesor.copy(other, stream: stream, context: context);

  @override
  I8COnesor read({Context? context, CudaStream? stream}) {
    final ret = I8COnesor.sized(length, context: context);
    stream = stream ?? CudaStream.noStream(deviceId);
    cuda.memcpy(stream, ret.ptr.cast(), ptr.cast(), ret.lengthBytes);
    return ret;
  }

  @override
  I8CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream.noStream(deviceId);
      final ret = I8CuOnesor.sized(stream, length, context: context);
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
  I8CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I8CuOnesorView(this, start, length);
  }
}

class _I8CuOnesor
    with Onesor<int>, ListMixin<int>, I8Onesor, CuOnesor<int>, I8CuOnesor
    implements I8CuOnesor {
  ffi.Pointer<ffi.Int8> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _I8CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _I8CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr = cuda.allocate(stream, length * Int8List.bytesPerElement);
    return _I8CuOnesor(ptr.cast(), length, stream.deviceId, context: context);
  }

  static _I8CuOnesor fromList(CudaStream stream, Int8List list,
      {Context? context}) {
    final ret = _I8CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(I8DartOnesor(list), stream: stream);
    return ret;
  }

  static _I8CuOnesor copy(Onesor<int> other,
      {CudaStream? stream, Context? context}) {
    stream = stream ?? CudaStream.noStream(other.deviceId);
    final ret = _I8CuOnesor.sized(stream, other.length, context: context);
    ret.copyFrom(other, stream: stream);
    return ret;
  }

  @override
  ffi.Pointer<ffi.Int8> get ptr => _ptr;

  @override
  void release({CudaStream? stream}) {
    if (_ptr == ffi.nullptr) return;
    stream ??= CudaStream.noStream(deviceId);
    cuda.memFree(stream, _ptr.cast());
    _ptr = ffi.nullptr;
  }
}

class I8CuOnesorView
    with Onesor<int>, I8Onesor, ListMixin<int>, CuOnesor<int>, I8CuOnesor
    implements I8CuOnesor, CuOnesorView<int>, I8OnesorView {
  final I8CuOnesor _list;

  @override
  final int offset;

  @override
  final int length;

  I8CuOnesorView(this._list, this.offset, this.length);

  @override
  int get deviceId => _list.deviceId;

  @override
  late final ffi.Pointer<ffi.Int8> ptr = _list.ptr.cast<ffi.Int8>() + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  I8CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I8CuOnesorView(_list, start + offset, length);
  }
}
