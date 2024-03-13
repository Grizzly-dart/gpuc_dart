import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';
import 'dart:io';
import 'package:path/path.dart' as path;

void initializeTensorC({String? libPath}) {
  String os;
  if (Platform.isLinux) {
    os = 'linux';
  } else if (Platform.isMacOS) {
    os = 'darwin';
  } else if (Platform.isWindows) {
    os = 'windows';
  } else {
    return;
  }

  String libraryPath;
  if (libPath != null) {
    libraryPath = libPath;
  } else {
    libraryPath = path.join(Directory.current.path, 'lib', 'asset', os);
  }
  if (Platform.isLinux) {
    libraryPath = path.join(libraryPath, 'libtensorc.so');
  } else if (Platform.isMacOS) {
    libraryPath = path.join(libraryPath, 'libtensorc.dylib');
  } else if (Platform.isWindows) {
    libraryPath = path.join(libraryPath, 'libtensorc.dll');
  } else {
    throw Exception('Unsupported platform');
  }

  final dylib = ffi.DynamicLibrary.open(libraryPath);
  CFFI.initialize(dylib);
}

typedef VoidPtr = ffi.Pointer<ffi.Void>;
typedef StrPtr = ffi.Pointer<ffi.Utf8>;
typedef F64Ptr = ffi.Pointer<ffi.Double>;

final class CSize2D extends ffi.Struct {
  @ffi.Uint32()
  external int r;

  @ffi.Uint32()
  external int c;

  static CPtr<CSize2D> allocate({Context? context}) =>
      CPtr<CSize2D>.allocate(ffi.sizeOf<CSize2D>(), context: context);

  static CPtr<CSize2D> fromSize2D(Dim2 size, {Context? context}) {
    final cptr = CSize2D.allocate(context: context);
    final cSize = cptr.ptr.ref;
    cSize.r = size.rows;
    cSize.c = size.cols;
    return cptr;
  }
}

final class CSize3D extends ffi.Struct {
  @ffi.Uint32()
  external int ch;

  @ffi.Uint32()
  external int r;

  @ffi.Uint32()
  external int c;

  static CPtr<CSize3D> allocate({Context? context}) =>
      CPtr<CSize3D>.allocate(ffi.sizeOf<CSize2D>(), context: context);

  static CPtr<CSize3D> fromSize(Dim size, {Context? context}) {
    final cptr = CSize3D.allocate(context: context);
    final cSize = cptr.ptr.ref;
    cSize.ch = size.channels;
    cSize.r = size.rows;
    cSize.c = size.cols;
    return cptr;
  }
}

class CPtr<T extends ffi.NativeType> implements Resource, ffi.Finalizable {
  ffi.Pointer<T> _mem;

  CPtr(this._mem, {Context? context}) {
    CFFI.finalizer.attach(this, _mem.cast());
    context?.add(this);
  }

  factory CPtr.allocate(int byteSizePerItem,
          {int count = 1, Context? context}) =>
      CPtr(ffi.malloc.allocate(byteSizePerItem * 1), context: context);

  ffi.Pointer<T> get ptr => _mem;

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

abstract class CFFI {
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

  static final finalizer = ffi.NativeFinalizer(CFFI.freeNative);
}
