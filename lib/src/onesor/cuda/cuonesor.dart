import 'dart:ffi' as ffi;

import 'package:gpuc_dart/gpuc_dart.dart';

export 'f64cuonesor.dart';

abstract mixin class CuOnesor<T extends num> implements Onesor<T> {
  factory CuOnesor.copy(CudaStream stream, Onesor<T> other,
      {Context? context}) {
    final type = other.numType;
    if (type == f64) {
      return F64CuOnesor.copy(stream, other as Onesor<double>, context: context)
          as CuOnesor<T>;
    }
    /* else if (type == NumType.f32) {
      return F32CuOnesor.copy(other as Onesor<double>,
          stream: stream, context: context) as CuOnesor<T>;
    } else if (type == NumType.i32) {
      return I32CuOnesor.copy(other as Onesor<int>,
          stream: stream, context: context) as CuOnesor<T>;
    } else if (type == NumType.i64) {
      return I64CuOnesor.copy(other as Onesor<int>,
          stream: stream, context: context) as CuOnesor<T>;
    }*/
    throw UnsupportedError('Unsupported number type: $type');
    // TODO
  }

  factory CuOnesor.sized(CudaStream stream, NumType<T> type, int length,
      {Context? context}) {
    if (type == f64) {
      return F64CuOnesor.sized(stream, length, context: context) as CuOnesor<T>;
    }
    // TODO
    throw UnsupportedError('Unsupported number type: $type');
  }

  ffi.Pointer<ffi.SizedNativeType> get ptr;

  @override
  DeviceType get deviceType => DeviceType.cuda;

  @override
  COnesor<T> read({Context? context, CudaStream? stream});

  @override
  CuOnesor<T> slice(int start, int length,
      {Context? context, CudaStream? stream});

  @override
  void release({CudaStream? stream});

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

  @override
  set length(int newLength) {
    throw UnsupportedError('Length cannot be changed');
  }
}

abstract class CuOnesorView<T extends num>
    implements CuOnesor<T>, OnesorView<T> {}
