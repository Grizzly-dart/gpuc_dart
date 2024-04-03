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

  late final finalizer = Finalizer((other) {
    if (other is ffi.Pointer) {
      free(other.cast());
    } else {
      stdout.writeln('Unknown type ${other.runtimeType}');
    }
  });
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

  int bytes;

  CPtr(this._mem, this.bytes, {Context? context}) {
    assert(_mem != ffi.nullptr);
    cffi!.finalizer.attach(this, _mem.cast(), detach: this);
    context?.add(this);
  }

  factory CPtr.allocate(int byteSizePerItem,
          {int count = 1, Context? context}) =>
      CPtr(
          ffi.malloc.allocate(byteSizePerItem * count), byteSizePerItem * count,
          context: context);

  ffi.Pointer<T> get ptr => _mem;

  void realloc(int byteSizePerItem, {int count = 1}) {
    final newPtr = cffi!.realloc(_mem.cast(), byteSizePerItem * count);
    if (newPtr == ffi.nullptr) {
      throw Exception('Failed to allocate memory');
    }
    _mem = newPtr.cast();
    bytes = byteSizePerItem * count;
    cffi!.finalizer.detach(this);
    cffi!.finalizer.attach(this, _mem.cast(), detach: this);
  }

  @override
  void release() {
    if (_mem == ffi.nullptr) return;
    cffi!.finalizer.detach(_mem);
    ffi.malloc.free(_mem);
    _mem = ffi.nullptr;
  }

  @override
  void coRelease(Resource other) {
    cffi!.finalizer.attach(this, other, detach: other);
  }

  @override
  void detachCoRelease(Resource other) {
    cffi!.finalizer.detach(other);
  }
}

class NumType<T extends num> {
  final int id;
  final String name;
  final String short;
  final Type ffiType;
  final T defaultVal;
  final T minVal;
  final T maxVal;
  final int bytes;

  const NumType._(this.name, this.id, this.short, this.ffiType, this.defaultVal,
      this.minVal, this.maxVal, this.bytes);

  CPtr allocate(int length) => CPtr.allocate(bytes, count: length);

  CPtr allocateForValue(num value) {
    final ptr = CPtr.allocate(bytes);
    if (ffiType == ffi.Double) {
      ptr.ptr.cast<ffi.Double>().value = value.toDouble();
    } else if (ffiType == ffi.Float) {
      ptr.ptr.cast<ffi.Float>().value = value.toDouble();
    } else if (ffiType == ffi.Int8) {
      ptr.ptr.cast<ffi.Int8>().value = value.toInt();
    } else if (ffiType == ffi.Int16) {
      ptr.ptr.cast<ffi.Int16>().value = value.toInt();
    } else if (ffiType == ffi.Int32) {
      ptr.ptr.cast<ffi.Int32>().value = value.toInt();
    } else if (ffiType == ffi.Int64) {
      ptr.ptr.cast<ffi.Int64>().value = value.toInt();
    } else if (ffiType == ffi.Uint8) {
      ptr.ptr.cast<ffi.Uint8>().value = value.toInt();
    } else if (ffiType == ffi.Uint16) {
      ptr.ptr.cast<ffi.Uint16>().value = value.toInt();
    } else if (ffiType == ffi.Uint32) {
      ptr.ptr.cast<ffi.Uint32>().value = value.toInt();
    } else if (ffiType == ffi.Uint64) {
      ptr.ptr.cast<ffi.Uint64>().value = value.toInt();
    } else {
      throw Exception('Unknown type $ffiType');
    }
    return ptr;
  }

  CPtr allocateForList(Iterable<num> array) {
    final ptr = CPtr.allocate(bytes, count: array.length);
    if (ffiType == ffi.Double) {
      ptr.ptr
          .cast<ffi.Double>()
          .asTypedList(array.length)
          .setAll(0, array.map((e) => e.toDouble()));
    } else if (ffiType == ffi.Float) {
      ptr.ptr
          .cast<ffi.Float>()
          .asTypedList(array.length)
          .setAll(0, array.map((e) => e.toDouble()));
    } else if (ffiType == ffi.Int8) {
      ptr.ptr
          .cast<ffi.Int8>()
          .asTypedList(array.length)
          .setAll(0, array.map((e) => e.toInt()));
    } else if (ffiType == ffi.Int16) {
      ptr.ptr
          .cast<ffi.Int16>()
          .asTypedList(array.length)
          .setAll(0, array.map((e) => e.toInt()));
    } else if (ffiType == ffi.Int32) {
      ptr.ptr
          .cast<ffi.Int32>()
          .asTypedList(array.length)
          .setAll(0, array.map((e) => e.toInt()));
    } else if (ffiType == ffi.Int64) {
      ptr.ptr
          .cast<ffi.Int64>()
          .asTypedList(array.length)
          .setAll(0, array.map((e) => e.toInt()));
    } else if (ffiType == ffi.Uint8) {
      ptr.ptr
          .cast<ffi.Uint8>()
          .asTypedList(array.length)
          .setAll(0, array.map((e) => e.toInt()));
    } else if (ffiType == ffi.Uint16) {
      ptr.ptr
          .cast<ffi.Uint16>()
          .asTypedList(array.length)
          .setAll(0, array.map((e) => e.toInt()));
    } else if (ffiType == ffi.Uint32) {
      ptr.ptr
          .cast<ffi.Uint32>()
          .asTypedList(array.length)
          .setAll(0, array.map((e) => e.toInt()));
    } else if (ffiType == ffi.Uint64) {
      ptr.ptr
          .cast<ffi.Uint64>()
          .asTypedList(array.length)
          .setAll(0, array.map((e) => e.toInt()));
    } else {
      throw Exception('Unknown type $ffiType');
    }
    return ptr;
  }

  T get(ffi.Pointer ptr, {int index = 0}) {
    if (ffiType == ffi.Double) {
      return ptr.pointerAddition(index, bytes).cast<ffi.Double>().value as T;
    } else if (ffiType == ffi.Float) {
      return ptr.pointerAddition(index, bytes).cast<ffi.Float>().value as T;
    } else if (ffiType == ffi.Int8) {
      return ptr.pointerAddition(index, bytes).cast<ffi.Int8>().value as T;
    } else if (ffiType == ffi.Int16) {
      return ptr.pointerAddition(index, bytes).cast<ffi.Int16>().value as T;
    } else if (ffiType == ffi.Int32) {
      return ptr.pointerAddition(index, bytes).cast<ffi.Int32>().value as T;
    } else if (ffiType == ffi.Int64) {
      return ptr.pointerAddition(index, bytes).cast<ffi.Int64>().value as T;
    } else if (ffiType == ffi.Uint8) {
      return ptr.pointerAddition(index, bytes).cast<ffi.Uint8>().value as T;
    } else if (ffiType == ffi.Uint16) {
      return ptr.pointerAddition(index, bytes).cast<ffi.Uint16>().value as T;
    } else if (ffiType == ffi.Uint32) {
      return ptr.pointerAddition(index, bytes).cast<ffi.Uint32>().value as T;
    } else if (ffiType == ffi.Uint64) {
      return ptr.pointerAddition(index, bytes).cast<ffi.Uint64>().value as T;
    } else {
      throw Exception('Unknown type $ffiType');
    }
  }

  bool get isSInt =>
      ffiType is ffi.Int8 ||
      ffiType is ffi.Int16 ||
      ffiType is ffi.Int32 ||
      ffiType is ffi.Int64;

  bool get isUInt =>
      ffiType is ffi.Uint8 ||
      ffiType is ffi.Uint16 ||
      ffiType is ffi.Uint32 ||
      ffiType is ffi.Uint64;

  bool get isXInt => isSInt || isUInt;

  bool get isFloat => ffiType is ffi.Float || ffiType is ffi.Double;

  static const List<NumType> values = [
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    f32,
    f64,
  ];

  static NumType typeOf(ffi.Pointer ptr) {
    if (ptr is ffi.Float) {
      return f32;
    } else if (ptr is ffi.Double) {
      return f64;
    } else if (ptr is ffi.Int8) {
      return i8;
    } else if (ptr is ffi.Int16) {
      return i16;
    } else if (ptr is ffi.Int32) {
      return i32;
    } else if (ptr is ffi.Int64) {
      return i64;
    } else if (ptr is ffi.Uint8) {
      return u8;
    } else if (ptr is ffi.Uint16) {
      return u16;
    } else if (ptr is ffi.Uint32) {
      return u32;
    } else if (ptr is ffi.Uint64) {
      return u64;
    } else {
      throw Exception('Unknown type ${ptr.runtimeType}');
    }
  }
}

const NumType<int> i8 = NumType._('int8', 1, 'i8', ffi.Int8, 0, -128, 127, 1);
const NumType<int> i16 =
    NumType._('int16', 2, 'i16', ffi.Int16, 0, -32768, 32767, 2);
const NumType<int> i32 =
    NumType._('int32', 4, 'i32', ffi.Int32, 0, -2147483648, 2147483647, 4);
const NumType<int> i64 = NumType._('int64', 8, 'i64', ffi.Int64, 0,
    -9223372036854775808, 9223372036854775807, 8);

const NumType<int> u8 = NumType._('uint8', 17, 'u8', ffi.Uint8, 0, 0, 255, 1);
const NumType<int> u16 =
    NumType._('uint16', 18, 'u16', ffi.Uint16, 0, 0, 65535, 2);
const NumType<int> u32 =
    NumType._('uint32', 20, 'u32', ffi.Uint32, 0, 0, 4294967295, 4);
const NumType<int> u64 =
    NumType._('uint64', 24, 'u64', ffi.Uint64, 0, 0, 9223372036854775807, 8);

const NumType<double> f32 = NumType._('float32', 36, 'f32', ffi.Float, 0.0,
    double.negativeInfinity, double.infinity, 4);
const NumType<double> f64 = NumType._('float64', 40, 'f64', ffi.Double, 0.0,
    double.negativeInfinity, double.infinity, 8);

extension PointerExt<T extends ffi.NativeType> on ffi.Pointer<T> {
  ffi.Pointer<T> pointerAddition(int offset, int bytesPerItem) =>
      ffi.Pointer.fromAddress(address + offset * bytesPerItem);
}
