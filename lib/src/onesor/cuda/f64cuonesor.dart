import 'dart:collection';
import 'dart:ffi' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract class CuOnesor<T extends num> implements Onesor<T> {
  ffi.Pointer<ffi.SizedNativeType> get ptr;

  @override
  COnesor<T> read({Context? context, CudaStream? stream});

  @override
  void copyFrom(Onesor<T> src, {CudaStream? stream});

  @override
  void copyTo(Onesor<T> dst, {CudaStream? stream});

  @override
  CuOnesor<T> slice(int start, int length,
      {Context? context, CudaStream? stream});

  @override
  void release({CudaStream? stream});
}

mixin CuOnesorMixin<T extends num> implements CuOnesor<T> {
  @override
  void copyFrom(Onesor<T> src, {CudaStream? stream}) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    final context = Context();
    try {
      stream = stream ?? CudaStream(deviceId, context: context);
      src = src is COnesor<T> ? src : src.read(context: context);
      cuda.memcpy(stream, ptr.cast(), src.ptr.cast(), lengthBytes);
    } finally {
      context.release();
    }
  }

  @override
  void copyTo(Onesor<T> dst, {CudaStream? stream}) {
    if (lengthBytes != dst.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    final context = Context();
    stream = stream ?? CudaStream(deviceId, context: context);
    try {
      if (dst is COnesor<T>) {
        cuda.memcpy(stream, dst.ptr.cast(), ptr.cast(), dst.lengthBytes);
        return;
      }
      final cSrc = read(context: context, stream: stream);
      dst.copyFrom(cSrc);
    } finally {
      context.release();
    }
  }
}

abstract class F64CuOnesor implements Onesor<double>, CuOnesor<double> {
  @override
  ffi.Pointer<ffi.Double> get ptr;

  static const sizeOfItem = 8;

  factory F64CuOnesor(ffi.Pointer<ffi.Double> ptr, int length, int deviceId,
          {Context? context}) =>
      _F64CuOnesor(ptr, length, deviceId, context: context);

  static F64CuOnesor sized(CudaStream stream, int length, {Context? context}) =>
      _F64CuOnesor.sized(stream, length, context: context);

  static F64CuOnesor fromList(CudaStream stream, List<double> list,
          {Context? context}) =>
      _F64CuOnesor.fromList(stream, list, context: context);

  static F64CuOnesor copy(Onesor<double> other,
          {CudaStream? stream, Context? context}) =>
      _F64CuOnesor.copy(other, stream: stream, context: context);
}

class _F64CuOnesor
    with CuOnesorMixin<double>, F64CuOnesorMixin, ListMixin<double>
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
    final ptr = cuda.allocate(stream, length * F64CuOnesor.sizeOfItem);
    return _F64CuOnesor(ptr.cast(), length, stream.deviceId, context: context);
  }

  static _F64CuOnesor fromList(CudaStream stream, List<double> list,
      {Context? context}) {
    final ret = _F64CuOnesor.sized(stream, list.length, context: context);
    ret.copyFrom(DartOnesor<double>(list), stream: stream);
    return ret;
  }

  static _F64CuOnesor copy(Onesor<double> other,
      {CudaStream? stream, Context? context}) {
    final lContext = Context();
    try {
      stream = stream ?? CudaStream(other.deviceId, context: lContext);
      final ret = _F64CuOnesor.sized(stream, other.length, context: context);
      ret.copyFrom(other, stream: stream);
      return ret;
    } finally {
      lContext.release();
    }
  }

  @override
  ffi.Pointer<ffi.Double> get ptr => _ptr;

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

class F64CuOnesorView
    with CuOnesorMixin<double>, F64CuOnesorMixin, ListMixin<double>
    implements F64CuOnesor, OnesorView<double> {
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
}

mixin F64CuOnesorMixin implements F64CuOnesor {
  @override
  DeviceType get deviceType => DeviceType.cuda;

  @override
  int get lengthBytes => length * bytesPerItem;

  @override
  int get bytesPerItem => 8;

  @override
  double get defaultValue => 0;

  // TODO

  @override
  double operator [](int index) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    return cuda.getDouble(ptr, index, deviceId);
  }

  @override
  void operator []=(int index, double value) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    cuda.setDouble(ptr, index, value, deviceId);
  }

  @override
  set length(int newLength) {
    throw UnsupportedError('Length cannot be changed');
  }

  @override
  COnesor<double> read({Context? context, CudaStream? stream}) {
    final clist = F64COnesor.sized(length, context: context);
    final lContext = Context();
    try {
      stream = stream ?? CudaStream(deviceId, context: lContext);
      cuda.memcpy(stream, clist.ptr.cast(), ptr.cast(), clist.lengthBytes);
      return clist;
    } finally {
      lContext.release();
    }
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
      stream ??= CudaStream(deviceId, context: lContext);
      final ret = F64CuOnesor.sized(stream, length, context: context);
      lContext.releaseOnErr(ret);
      cuda.memcpy(stream, ret.ptr.cast(), (ptr + F64CuOnesor.sizeOfItem).cast(),
          length * F64CuOnesor.sizeOfItem);
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
    if (this is F64CuOnesorView) {
      return F64CuOnesorView((this as F64CuOnesorView)._list,
          start + (this as F64CuOnesorView).offset, length);
    }
    return F64CuOnesorView(this, start, length);
  }
}
