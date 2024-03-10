import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

final class CSize2D extends ffi.Struct {
  @ffi.Uint32()
  external int r;

  @ffi.Uint32()
  external int c;

  static ffi.Pointer<CSize2D> fromSize2D(Dim2 size, {Context? context}) {
    final cSize = ffi.malloc.allocate<CSize2D>(ffi.sizeOf<CSize2D>());
    CPtr(cSize.cast(), context: context);
    cSize.ref.r = size.rows;
    cSize.ref.c = size.cols;
    return cSize;
  }
}

class CPtr implements Resource {
  ffi.Pointer<ffi.Void> _mem;

  CPtr(this._mem, {Context? context}) {
    context?.add(this);
  }

  ffi.Pointer<ffi.Void> get ptr => _mem;

  @override
  void release() {
    if (_mem == ffi.nullptr) {
      return;
    }
    ffi.malloc.free(_mem);
    _mem = ffi.nullptr;
  }
}

class CF64Ptr implements Resource {
  ffi.Pointer<ffi.Double> _mem;

  CF64Ptr(this._mem, {Context? context}) {
    context?.add(this);
  }

  static CF64Ptr allocate({Context? context}) {
    final mem = ffi.calloc<ffi.Double>(8);
    return CF64Ptr(mem, context: context);
  }

  ffi.Pointer<ffi.Double> get ptr => _mem;

  double get value => _mem.value;

  set value(double value) => _mem.value = value;

  @override
  void release() {
    if (_mem == ffi.nullptr) {
      return;
    }
    ffi.malloc.free(_mem);
    _mem = ffi.nullptr;
  }
}

abstract class CListFFI {
  static late final ffi
      .Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>
  freeNative;
  static late final void Function(ffi.Pointer<ffi.Void>) free;
  static late final ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void> oldPtr, int size) realloc;
  static late final void Function(
      ffi.Pointer<ffi.Void> dst, ffi.Pointer<ffi.Void> src, int size) memcpy;

  static void initialize(ffi.DynamicLibrary dylib) {
    freeNative = dylib
        .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
        'libtcFree');
    free = dylib.lookupFunction<ffi.Void Function(ffi.Pointer<ffi.Void>),
        void Function(ffi.Pointer<ffi.Void>)>('libtcFree');
    realloc = dylib.lookupFunction<
        ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Uint64),
        ffi.Pointer<ffi.Void> Function(
            ffi.Pointer<ffi.Void>, int)>('libtcRealloc');
    memcpy = dylib.lookupFunction<
        ffi.Void Function(
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, ffi.Uint64),
        void Function(
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, int)>('libtcMemcpy');
  }

  static final finalizer = ffi.NativeFinalizer(CListFFI.freeNative);
}
