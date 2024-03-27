import 'dart:ffi' as ffi;

import 'package:gpuc_dart/gpuc_dart.dart';

export 'f64conesor.dart';
export 'f32conesor.dart';
export 'i64conesor.dart';
export 'u64conesor.dart';
export 'i32conesor.dart';
export 'u32conesor.dart';
export 'i16conesor.dart';
export 'u16conesor.dart';
export 'i8conesor.dart';
export 'u8conesor.dart';

abstract mixin class COnesor<T extends num> implements Onesor<T>, NumPtr {
  @override
  DeviceType get deviceType => DeviceType.c;

  @override
  int get deviceId => 0;

  @override
  ffi.Pointer<ffi.SizedNativeType> get ptr;

  List<T> asTypedList(int length);

  factory COnesor.sized(int length, {Context? context}) {
    if(T == double) {
      return F64COnesor.sized(length, context: context) as COnesor<T>;
    }
    throw UnimplementedError('Type $T not implemented');
  }

  @override
  void copyFrom(Onesor<T> src) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is COnesor<T>) {
      cffi!.memcpy(ptr.cast(), src.ptr.cast(), lengthBytes);
    } else if (src is DartOnesor<T>) {
      asTypedList(length).setAll(0, src);
    }
    src.copyTo(this);
  }

  @override
  void copyTo(Onesor<T> dst) {
    if (lengthBytes != dst.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (dst is COnesor<T>) {
      cffi!.memcpy(dst.ptr.cast(), ptr.cast(), lengthBytes);
      return;
    } else if (dst is DartOnesor<T>) {
      dst.setAll(0, asTypedList(length));
      return;
    }
    dst.copyFrom(this);
  }
}

abstract class COnesorView<T extends num>
    implements COnesor<T>, OnesorView<T> {}
