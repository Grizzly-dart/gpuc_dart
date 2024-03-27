import 'dart:ffi' as ffi;

import 'package:gpuc_dart/gpuc_dart.dart';

export 'f64cuonesor.dart';
export 'f32cuonesor.dart';
export 'u64cuonesor.dart';
export 'u32cuonesor.dart';
export 'u16cuonesor.dart';
export 'u8cuonesor.dart';
export 'i64cuonesor.dart';
export 'i32cuonesor.dart';
export 'i16cuonesor.dart';
export 'i8cuonesor.dart';

abstract mixin class CuOnesor<T extends num> implements Onesor<T>, NumPtr {
  factory CuOnesor.copy(CudaStream stream, Onesor<T> other,
      {Context? context}) {
    final type = other.type;
    if (type == f64) {
      return F64CuOnesor.copy(stream, other as Onesor<double>, context: context)
          as CuOnesor<T>;
    } else if (type == f32) {
      return F32CuOnesor.copy(stream, other as Onesor<double>, context: context)
          as CuOnesor<T>;
    } else if (type == u64) {
      return U64CuOnesor.copy(stream, other as Onesor<int>, context: context)
          as CuOnesor<T>;
    } else if (type == u32) {
      return U32CuOnesor.copy(stream, other as Onesor<int>, context: context)
          as CuOnesor<T>;
    } else if (type == u16) {
      return U16CuOnesor.copy(stream, other as Onesor<int>, context: context)
          as CuOnesor<T>;
    } else if (type == u8) {
      return U8CuOnesor.copy(stream, other as Onesor<int>, context: context)
          as CuOnesor<T>;
    } else if (type == i64) {
      return I64CuOnesor.copy(stream, other as Onesor<int>, context: context)
          as CuOnesor<T>;
    } else if (type == i32) {
      return I32CuOnesor.copy(stream, other as Onesor<int>, context: context)
          as CuOnesor<T>;
    } else if (type == i16) {
      return I16CuOnesor.copy(stream, other as Onesor<int>, context: context)
          as CuOnesor<T>;
    } else if (type == i8) {
      return I8CuOnesor.copy(stream, other as Onesor<int>, context: context)
          as CuOnesor<T>;
    }
    throw UnsupportedError('Unsupported number type: $type');
  }

  factory CuOnesor.sized(CudaStream stream, NumType<T> type, int length,
      {Context? context}) {
    if (type == f64) {
      return F64CuOnesor.sized(stream, length, context: context) as CuOnesor<T>;
    } else if (type == f32) {
      return F32CuOnesor.sized(stream, length, context: context) as CuOnesor<T>;
    } else if (type == u64) {
      return U64CuOnesor.sized(stream, length, context: context) as CuOnesor<T>;
    } else if (type == u32) {
      return U32CuOnesor.sized(stream, length, context: context) as CuOnesor<T>;
    } else if (type == u16) {
      return U16CuOnesor.sized(stream, length, context: context) as CuOnesor<T>;
    } else if (type == u8) {
      return U8CuOnesor.sized(stream, length, context: context) as CuOnesor<T>;
    } else if (type == i64) {
      return I64CuOnesor.sized(stream, length, context: context) as CuOnesor<T>;
    } else if (type == i32) {
      return I32CuOnesor.sized(stream, length, context: context) as CuOnesor<T>;
    } else if (type == i16) {
      return I16CuOnesor.sized(stream, length, context: context) as CuOnesor<T>;
    } else if (type == i8) {
      return I8CuOnesor.sized(stream, length, context: context) as CuOnesor<T>;
    }
    throw UnsupportedError('Unsupported number type: $type');
  }

  @override
  ffi.Pointer<ffi.SizedNativeType> get ptr;

  @override
  DeviceType get deviceType => DeviceType.cuda;

  @override
  COnesor<T> read({Context? context, CudaStream? stream});

  /*
  @override
  CuOnesor<T> slice(int start, int length,
      {Context? context, CudaStream? stream});
   */

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
