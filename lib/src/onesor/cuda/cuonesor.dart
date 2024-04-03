import 'dart:ffi' as ffi;

import 'package:gpuc_dart/gpuc_dart.dart';
import 'dart:collection';
import 'dart:typed_data';

part 'f64cuonesor.dart';
part 'f32cuonesor.dart';
part 'u64cuonesor.dart';
part 'u32cuonesor.dart';
part 'u16cuonesor.dart';
part 'u8cuonesor.dart';
part 'i64cuonesor.dart';
part 'i32cuonesor.dart';
part 'i16cuonesor.dart';
part 'i8cuonesor.dart';

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

  @override
  T operator [](int index) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    return cuda.getOne(CudaStream.noStream(deviceId), ptr, type, index: index);
  }

  @override
  void operator []=(int index, T value) {
    if (index < 0 || index >= length) {
      throw RangeError('Index out of range');
    }
    cuda.setOne(CudaStream.noStream(deviceId), ptr, value, type, index: index);
  }

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

    stream = stream ?? CudaStream.noStream(deviceId);
    src = src is COnesor<T> ? src : src.read(); // TODO release with corelease
    cuda.memcpy(stream, ptr.cast(), src.ptr.cast(), lengthBytes);
  }

  @override
  void copyTo(Onesor<T> dst, {CudaStream? stream}) {
    if (lengthBytes != dst.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    final context = Context();
    stream = stream ?? CudaStream.noStream(deviceId);
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

mixin _CuOnesorMixin<T extends num> implements CuOnesor<T> {
  CuPtr<ffi.SizedNativeType> get _ptr;

  @override
  void release({CudaStream? stream}) {
    _ptr.release(stream: stream);
  }

  @override
  void coRelease(Resource other) {
    _ptr.coRelease(other);
  }

  @override
  void detachCoRelease(Resource other) {
    _ptr.detachCoRelease(other);
  }
}

abstract class CuOnesorView<T extends num>
    implements CuOnesor<T>, OnesorView<T> {}

mixin _CuOnesorViewMixin<T extends num> implements CuOnesorView<T> {
  CuOnesor<T> get _inner;

  @override
  void coRelease(Resource other) {
    _inner.coRelease(other);
  }

  @override
  void detachCoRelease(Resource other) {
    _inner.detachCoRelease(other);
  }
}
