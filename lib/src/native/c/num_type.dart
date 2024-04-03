import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

class NumPtr {
  final Ptr ptr;
  final NumType type;

  NumPtr(this.ptr, this.type);
}

class CPtr<T extends ffi.NativeType> implements Resource, ffi.Finalizable {
  ffi.Pointer<T> _mem;

  int bytes;

  CPtr(this._mem, this.bytes, {Context? context}) {
    assert(_mem != ffi.nullptr);
    TensorCFFI.finalizer.attach(this, _mem.cast(), detach: this);
    context?.add(this);
  }

  factory CPtr.allocate(int byteSizePerItem,
          {int count = 1, Context? context}) =>
      CPtr(
          ffi.malloc.allocate(byteSizePerItem * count), byteSizePerItem * count,
          context: context);

  ffi.Pointer<T> get ptr => _mem;

  void realloc(int byteSizePerItem, {int count = 1}) {
    _mem = tc.realloc(_mem.cast(), byteSizePerItem * count);
    bytes = byteSizePerItem * count;
    TensorCFFI.finalizer.detach(this);
    TensorCFFI.finalizer.attach(this, _mem.cast(), detach: this);
  }

  @override
  void release() {
    if (_mem == ffi.nullptr) return;
    TensorCFFI.finalizer.detach(_mem);
    ffi.malloc.free(_mem);
    _mem = ffi.nullptr;
  }

  @override
  void coRelease(Resource other) {
    TensorCFFI.finalizer.attach(this, other, detach: other);
  }

  @override
  void detachCoRelease(Resource other) {
    TensorCFFI.finalizer.detach(other);
  }
}

class NumType<T extends num> {
  final int id;
  final int index;
  final String name;
  final String short;
  final Type ffiType;
  final T defaultVal;
  final T minVal;
  final T maxVal;
  final int bytes;

  const NumType._(this.name, this.id, this.index, this.short, this.ffiType,
      this.defaultVal, this.minVal, this.maxVal, this.bytes);

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

const NumType<int> i8 =
    NumType._('int8', 1, 0, 'i8', ffi.Int8, 0, -128, 127, 1);
const NumType<int> i16 =
    NumType._('int16', 2, 1, 'i16', ffi.Int16, 0, -32768, 32767, 2);
const NumType<int> i32 =
    NumType._('int32', 4, 2, 'i32', ffi.Int32, 0, -2147483648, 2147483647, 4);
const NumType<int> i64 = NumType._('int64', 8, 3, 'i64', ffi.Int64, 0,
    -9223372036854775808, 9223372036854775807, 8);

const NumType<int> u8 =
    NumType._('uint8', 17, 4, 'u8', ffi.Uint8, 0, 0, 255, 1);
const NumType<int> u16 =
    NumType._('uint16', 18, 5, 'u16', ffi.Uint16, 0, 0, 65535, 2);
const NumType<int> u32 =
    NumType._('uint32', 20, 6, 'u32', ffi.Uint32, 0, 0, 4294967295, 4);
const NumType<int> u64 =
    NumType._('uint64', 24, 7, 'u64', ffi.Uint64, 0, 0, 9223372036854775807, 8);

const NumType<double> f32 = NumType._('float32', 36, 8, 'f32', ffi.Float, 0.0,
    double.negativeInfinity, double.infinity, 4);
const NumType<double> f64 = NumType._('float64', 40, 9, 'f64', ffi.Double, 0.0,
    double.negativeInfinity, double.infinity, 8);

typedef Ptr = ffi.Pointer;
typedef NativePtr = ffi.Pointer<ffi.SizedNativeType>;
typedef VoidPtr = ffi.Pointer<ffi.Void>;
typedef StrPtr = ffi.Pointer<ffi.Utf8>;
typedef F64Ptr = ffi.Pointer<ffi.Double>;

extension PointerExt<T extends ffi.NativeType> on ffi.Pointer<T> {
  ffi.Pointer<T> pointerAddition(int offset, int bytesPerItem) =>
      ffi.Pointer.fromAddress(address + offset * bytesPerItem);
}
