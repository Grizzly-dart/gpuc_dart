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
    return; // TODO windows not supported yet
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
    libraryPath = path.join(libraryPath, 'libtensorc.so');
  } else if (Platform.isWindows) {
    libraryPath = path.join(libraryPath, 'libtensorc.dll');
  } else {
    throw Exception('Unsupported platform');
  }

  final dylib = ffi.DynamicLibrary.open(libraryPath);
  CFFI.initialize(dylib);
}

CFFI? cffi;

class CFFI {
  final ffi
      .Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>
      freeNative;
  final void Function(ffi.Pointer<ffi.Void>) free;
  final ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void> oldPtr, int size)
      realloc;
  final void Function(
      ffi.Pointer<ffi.Void> dst, ffi.Pointer<ffi.Void> src, int size) memcpy;

  CFFI(
      {required this.freeNative,
      required this.free,
      required this.realloc,
      required this.memcpy});

  factory CFFI.lookup(ffi.DynamicLibrary dylib) {
    final freeNative = dylib
        .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            'libtcFree');
    final free = dylib.lookupFunction<ffi.Void Function(ffi.Pointer<ffi.Void>),
        void Function(ffi.Pointer<ffi.Void>)>('libtcFree');
    final realloc = dylib.lookupFunction<
        ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Uint64),
        ffi.Pointer<ffi.Void> Function(
            ffi.Pointer<ffi.Void>, int)>('libtcRealloc');
    final memcpy = dylib.lookupFunction<
        ffi.Void Function(
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, ffi.Uint64),
        void Function(
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, int)>('libtcMemcpy');

    return CFFI(
        freeNative: freeNative, free: free, realloc: realloc, memcpy: memcpy);
  }

  static void initialize(ffi.DynamicLibrary dylib) {
    cffi = CFFI.lookup(dylib);
  }

  late final finalizer = ffi.NativeFinalizer(freeNative);
}

typedef Ptr = ffi.Pointer;
typedef NativePtr = ffi.Pointer<ffi.SizedNativeType>;
typedef VoidPtr = ffi.Pointer<ffi.Void>;
typedef StrPtr = ffi.Pointer<ffi.Utf8>;
typedef F64Ptr = ffi.Pointer<ffi.Double>;

class NumPtr {
  final Ptr ptr;
  final NumType type;

  NumPtr(this.ptr, this.type);
}

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

class CPtr<T extends ffi.NativeType> implements Resource, ffi.Finalizable {
  ffi.Pointer<T> _mem;

  CPtr(this._mem, {Context? context}) {
    cffi!.finalizer.attach(this, _mem.cast());
    context?.add(this);
  }

  factory CPtr.allocate(int byteSizePerItem,
          {int count = 1, Context? context}) =>
      CPtr(ffi.malloc.allocate(byteSizePerItem * count), context: context);

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
