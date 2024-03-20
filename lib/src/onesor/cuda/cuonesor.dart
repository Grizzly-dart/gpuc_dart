import 'dart:ffi' as ffi;

import 'package:gpuc_dart/gpuc_dart.dart';

export 'f64cuonesor.dart';

abstract class CuOnesor<T extends num> implements Onesor<T> {
  factory CuOnesor.copy(Onesor<T> other,
      {CudaStream? stream, Context? context}) {
    if(other is F64Onesor) {
      // TODO
    }
    // TODO
  }

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
