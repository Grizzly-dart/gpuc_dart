import 'dart:ffi' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

typedef CNumType = ffi.Uint8;

final class CDim2 extends ffi.Struct {
  @ffi.Uint32()
  external int r;

  @ffi.Uint32()
  external int c;

  static CPtr<CDim2> allocate() => CPtr<CDim2>.allocate(ffi.sizeOf<CDim2>());

  static CPtr<CDim2> from(Dim2 size) {
    final cptr = CDim2.allocate();
    final cSize = cptr.ptr.ref;
    cSize.r = size.rows;
    cSize.c = size.cols;
    return cptr;
  }
}

final class CDim3 extends ffi.Struct {
  @ffi.Uint32()
  external int ch;

  @ffi.Uint32()
  external int r;

  @ffi.Uint32()
  external int c;

  static CPtr<CDim3> allocate() => CPtr<CDim3>.allocate(ffi.sizeOf<CDim2>());

  static CPtr<CDim3> from(Dim3 size) {
    final cptr = CDim3.allocate();
    final cSize = cptr.ptr.ref;
    cSize.ch = size.channels;
    cSize.r = size.rows;
    cSize.c = size.cols;
    return cptr;
  }
}