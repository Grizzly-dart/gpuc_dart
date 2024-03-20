import 'dart:ffi' as ffi;

import 'package:gpuc_dart/gpuc_dart.dart';

export 'f64conesor.dart';

abstract class COnesor<T extends num> extends Onesor<T> {
  ffi.Pointer<ffi.SizedNativeType> get ptr;

  List<T> asTypedList(int length);

  factory COnesor.sized(int length, {Context? context}) {
    if(T == double) {
      return F64COnesor.sized(length, context: context) as COnesor<T>;
    }
    throw UnimplementedError('Type $T not implemented');
  }
}

abstract class COnesorView<T extends num>
    implements COnesor<T>, OnesorView<T> {}

mixin COnesorMixin<T extends num> implements COnesor<T> {
  @override
  ffi.Pointer<ffi.SizedNativeType> get ptr;

  @override
  void copyFrom(Onesor<T> src) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is COnesor<T>) {
      CFFI.memcpy(ptr.cast(), src.ptr.cast(), lengthBytes);
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
      CFFI.memcpy(dst.ptr.cast(), ptr.cast(), lengthBytes);
      return;
    } else if (dst is DartOnesor<T>) {
      dst.setAll(0, asTypedList(length));
      return;
    }
    dst.copyFrom(this);
  }
}
