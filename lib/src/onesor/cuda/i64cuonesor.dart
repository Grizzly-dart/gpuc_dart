import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I64CuOnesor implements CuOnesor<int>, I64Onesor {
  @override
  ffi.Pointer<ffi.Int64> get ptr;

  factory I64CuOnesor(ffi.Pointer<ffi.Int64> ptr, int length, int deviceId,
          {Context? context}) =>
      _I64CuOnesor(ptr, length, deviceId, context: context);

  static I64CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _I64CuOnesor.sized(stream, length, context: context);

  static I64CuOnesor fromList(CudaStream stream, Int64List list,
          {Context? context}) =>
      _I64CuOnesor.fromList(stream, list, context: context);

  static I64CuOnesor copy(CudaStream? stream, Onesor<int> other,
          {Context? context}) =>
      _I64CuOnesor.copy(other, stream: stream, context: context);

  @override
  I64COnesor read({Context? context, CudaStream? stream}) {
    final ret = I64COnesor.sized(length, context: context);
    stream = stream ?? CudaStream.noStream(deviceId);
    cuda.memcpy(stream, ret.ptr.cast(), ptr.cast(), ret.lengthBytes);
    return ret;
  }

  @override
  I64CuOnesor slice(int start, int length,
      {Context? context, CudaStream? stream}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final lContext = Context();
    try {
      stream ??= CudaStream.noStream(deviceId);
      final ret = I64CuOnesor.sized(stream, length, context: context);
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
  I64CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I64CuOnesorView(this, start, length);
  }
}

class _I64CuOnesor
    with Onesor<int>, ListMixin<int>, I64Onesor, CuOnesor<int>, I64CuOnesor
    implements I64CuOnesor {
  ffi.Pointer<ffi.Int64> _ptr;

  @override
  final int length;

  @override
  final int deviceId;

  _I64CuOnesor(this._ptr, this.length, this.deviceId, {Context? context}) {
    context?.add(this);
  }

  static _I64CuOnesor sized(CudaStream stream, int length, {Context? context}) {
    final ptr = cuda.allocate(stream, length * Int64List.bytesPerElement);
    return _I64CuOnesor(ptr.cast(), length, stream.deviceId, context: context);
  }

  static _I64CuOnesor fromList(CudaStream stream, Int64List list,
      {Context? context}) {
    final ret = _I64CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(I64DartOnesor(list), stream: stream);
    return ret;
  }

  static _I64CuOnesor copy(Onesor<int> other,
      {CudaStream? stream, Context? context}) {
    stream = stream ?? CudaStream.noStream(other.deviceId);
    final ret = _I64CuOnesor.sized(stream, other.length, context: context);
    ret.copyFrom(other, stream: stream);
    return ret;
  }

  @override
  ffi.Pointer<ffi.Int64> get ptr => _ptr;

  @override
  void release({CudaStream? stream}) {
    if (_ptr == ffi.nullptr) return;
    stream ??= CudaStream.noStream(deviceId);
    cuda.memFree(stream, _ptr.cast());
    _ptr = ffi.nullptr;
  }
}

class I64CuOnesorView
    with Onesor<int>, I64Onesor, ListMixin<int>, CuOnesor<int>, I64CuOnesor
    implements I64CuOnesor, CuOnesorView<int>, I64OnesorView {
  final I64CuOnesor _list;

  @override
  final int offset;

  @override
  final int length;

  I64CuOnesorView(this._list, this.offset, this.length);

  @override
  int get deviceId => _list.deviceId;

  @override
  late final ffi.Pointer<ffi.Int64> ptr = _list.ptr.cast<ffi.Int64>() + offset;

  @override
  void release({CudaStream? stream}) {}

  @override
  I64CuOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I64CuOnesorView(_list, start + offset, length);
  }
}
