import 'dart:ffi' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

export 'releaseable.dart';
export 'dart/dartonesor.dart';
export 'cuda/cuonesor.dart';
export 'c/conesor.dart';

// TODO complex onesor?
abstract mixin class Onesor<T extends num> implements Resource, List<T> {
  DeviceType get deviceType;

  int get deviceId;

  NumType<T> get type;

  int get lengthBytes => length * bytesPerItem;

  int get bytesPerItem;

  // TODO subview

  // TODO implement partial write
  void copyFrom(Onesor<T> src);

  void copyFromList(List<T> src) {
    if (length != src.length) {
      throw ArgumentError('Length mismatch');
    }
    setAll(0, src);
  }

  // TODO implement partial read
  void copyTo(Onesor<T> dst);

  COnesor<T> read({Context? context});

  OnesorView<T> view(int start, int length);

  Onesor<T> slice(int start, int length, {Context? context});

  T get defaultValue;

  Tensor<T> toTensor(Dim size, {String name = '', Context? context});

  @override
  void release();
}

abstract class OnesorView<T extends num> extends Onesor<T> {
  int get offset;
}

extension OnesorExtension<T extends num> on Onesor<T> {
  Device get device => Device(deviceType, deviceId);
}

abstract mixin class F64Onesor implements Onesor<double> {
  @override
  NumType<double> get type => f64;

  @override
  double get defaultValue => 0.0;

  @override
  int get bytesPerItem => 8;

  @override
  F64OnesorView view(int start, int length);

  @override
  F64Tensor toTensor(Dim size, {String name = '', Context? context}) =>
      F64Tensor(this, size, name: name, context: context);
}

abstract mixin class F64OnesorView implements F64Onesor, OnesorView<double> {}

abstract mixin class F32Onesor implements Onesor<double> {
  @override
  NumType<double> get type => f32;

  @override
  int get lengthBytes => length * bytesPerItem;

  @override
  double get defaultValue => 0.0;

  @override
  int get bytesPerItem => 4;

  @override
  F32OnesorView view(int start, int length);

  @override
  F32Tensor toTensor(Dim size, {String name = '', Context? context}) =>
      F32Tensor(this, size, name: name, context: context);
}

abstract mixin class F32OnesorView implements F32Onesor, OnesorView<double> {}

abstract mixin class U64Onesor implements Onesor<int> {
  @override
  NumType<int> get type => u64;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 8;

  @override
  U64Onesor slice(int start, int length, {Context? context});

  @override
  U64OnesorView view(int start, int length);

  @override
  U64Tensor toTensor(Dim size, {String name = '', Context? context}) =>
      U64Tensor(this, size, name: name, context: context);
}

abstract mixin class U64OnesorView implements U64Onesor, OnesorView<int> {}

abstract mixin class I64Onesor implements Onesor<int> {
  @override
  NumType<int> get type => i64;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 8;

  @override
  I64OnesorView view(int start, int length);

  @override
  I64Tensor toTensor(Dim size, {String name = '', Context? context}) =>
      I64Tensor(this, size, name: name, context: context);
}

abstract mixin class I64OnesorView implements I64Onesor, OnesorView<int> {}

abstract mixin class I32Onesor implements Onesor<int> {
  @override
  NumType<int> get type => i32;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 4;

  @override
  I32OnesorView view(int start, int length);

  @override
  I32Tensor toTensor(Dim size, {String name = '', Context? context}) =>
      I32Tensor(this, size, name: name, context: context);
}

abstract mixin class I32OnesorView implements I32Onesor, OnesorView<int> {}

abstract mixin class U32Onesor implements Onesor<int> {
  @override
  NumType<int> get type => u32;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 4;

  @override
  U32OnesorView view(int start, int length);

  @override
  U32Tensor toTensor(Dim size, {String name = '', Context? context}) =>
      U32Tensor(this, size, name: name, context: context);
}

abstract mixin class U32OnesorView implements U32Onesor, OnesorView<int> {}

abstract mixin class I16Onesor implements Onesor<int> {
  @override
  NumType<int> get type => i16;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 2;

  @override
  I16OnesorView view(int start, int length);

  @override
  I16Tensor toTensor(Dim size, {String name = '', Context? context}) =>
      I16Tensor(this, size, name: name, context: context);
}

abstract mixin class I16OnesorView implements I16Onesor, OnesorView<int> {}

abstract mixin class U16Onesor implements Onesor<int> {
  @override
  NumType<int> get type => u16;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 2;

  @override
  U16OnesorView view(int start, int length);

  @override
  U16Tensor toTensor(Dim size, {String name = '', Context? context}) =>
      U16Tensor(this, size, name: name, context: context);
}

abstract mixin class U16OnesorView implements U16Onesor, OnesorView<int> {}

abstract mixin class I8Onesor implements Onesor<int> {
  @override
  NumType<int> get type => i8;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 1;

  @override
  I8OnesorView view(int start, int length);

  @override
  I8Tensor toTensor(Dim size, {String name = '', Context? context}) =>
      I8Tensor(this, size, name: name, context: context);
}

abstract mixin class I8OnesorView implements I8Onesor, OnesorView<int> {}

abstract mixin class U8Onesor implements Onesor<int> {
  @override
  NumType<int> get type => u8;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 1;

  @override
  U8OnesorView view(int start, int length);

  @override
  U8Tensor toTensor(Dim size, {String name = '', Context? context}) =>
      U8Tensor(this, size, name: name, context: context);
}

abstract mixin class U8OnesorView implements U8Onesor, OnesorView<int> {}

enum DeviceType { c, dart, cuda, rocm, sycl }

class Device {
  final DeviceType type;
  final int id;

  Device(this.type, this.id);

  @override
  bool operator ==(Object other) {
    if (other is! Device) return false;
    if (identical(this, other)) return true;
    if (type != other.type) return false;
    if (type == DeviceType.c || type == DeviceType.dart) return true;
    return type == other.type && id == other.id;
  }

  @override
  int get hashCode => Object.hashAll([type.index, id]);
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

const NumType<int> i8 = NumType._('int8', 0, 'i8', ffi.Int8, 0, -128, 127, 1);
const NumType<int> i16 =
    NumType._('int16', 1, 'i16', ffi.Int16, 0, -32768, 32767, 2);
const NumType<int> i32 =
    NumType._('int32', 2, 'i32', ffi.Int32, 0, -2147483648, 2147483647, 4);
const NumType<int> i64 = NumType._('int64', 3, 'i64', ffi.Int64, 0,
    -9223372036854775808, 9223372036854775807, 8);

const NumType<int> u8 = NumType._('uint8', 10, 'u8', ffi.Uint8, 0, 0, 255, 1);
const NumType<int> u16 =
    NumType._('uint16', 11, 'u16', ffi.Uint16, 0, 0, 65535, 2);
const NumType<int> u32 =
    NumType._('uint32', 12, 'u32', ffi.Uint32, 0, 0, 4294967295, 4);
const NumType<int> u64 =
    NumType._('uint64', 13, 'u64', ffi.Uint64, 0, 0, 9223372036854775807, 8);

const NumType<double> f32 = NumType._('float32', 22, 'f32', ffi.Float, 0.0,
    double.negativeInfinity, double.infinity, 4);
const NumType<double> f64 = NumType._('float64', 23, 'f64', ffi.Double, 0.0,
    double.negativeInfinity, double.infinity, 8);
