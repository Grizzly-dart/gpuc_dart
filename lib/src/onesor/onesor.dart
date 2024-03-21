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

  NumType<T> get numType;

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

  Tensor<T> toTensor(Dim size, {Context? context});

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
  NumType<double> get numType => NumType.f64;

  @override
  double get defaultValue => 0.0;

  @override
  int get bytesPerItem => 8;

  @override
  F64Tensor toTensor(Dim size, {Context? context}) =>
      F64Tensor(this, size, context: context);
}

abstract mixin class F32Onesor implements Onesor<double> {
  @override
  NumType<double> get numType => NumType.f32;

  @override
  int get lengthBytes => length * bytesPerItem;

  @override
  double get defaultValue => 0.0;

  @override
  int get bytesPerItem => 4;


}

abstract mixin class I64Onesor implements Onesor<int> {
  @override
  NumType<int> get numType => NumType.i64;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 8;
}

abstract mixin class U64Onesor implements Onesor<int> {
  @override
  NumType<int> get numType => NumType.u64;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 8;
}

abstract mixin class I32Onesor implements Onesor<int> {
  @override
  NumType<int> get numType => NumType.i32;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 4;
}

abstract mixin class U32Onesor implements Onesor<int> {
  @override
  NumType<int> get numType => NumType.u32;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 4;
}

abstract mixin class I16Onesor implements Onesor<int> {
  @override
  NumType<int> get numType => NumType.i16;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 2;
}

abstract mixin class U16Onesor implements Onesor<int> {
  @override
  NumType<int> get numType => NumType.u16;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 2;
}

abstract mixin class I8Onesor implements Onesor<int> {
  @override
  NumType<int> get numType => NumType.i8;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 1;
}

abstract mixin class U8Onesor implements Onesor<int> {
  @override
  NumType<int> get numType => NumType.u8;

  @override
  int get defaultValue => 0;

  @override
  int get bytesPerItem => 1;
}

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

class NumType<T> {
  final int id;
  final String name;
  final ffi.SizedNativeType ffiType;

  const NumType._(this.name, this.id, this.ffiType);

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

  static NumType typeOf(ffi.Pointer ptr) {
    if (ptr is ffi.Float) {
      return NumType.f32;
    } else if (ptr is ffi.Double) {
      return NumType.f64;
    } else if (ptr is ffi.Int8) {
      return NumType.i8;
    } else if (ptr is ffi.Int16) {
      return NumType.i16;
    } else if (ptr is ffi.Int32) {
      return NumType.i32;
    } else if (ptr is ffi.Int64) {
      return NumType.i64;
    } else if (ptr is ffi.Uint8) {
      return NumType.u8;
    } else if (ptr is ffi.Uint16) {
      return NumType.u16;
    } else if (ptr is ffi.Uint32) {
      return NumType.u32;
    } else if (ptr is ffi.Uint64) {
      return NumType.u64;
    } else {
      throw Exception('Unknown type');
    }
  }

  static const NumType<int> i8 = NumType._('int8', 0, ffi.Int8());
  static const NumType<int> i16 = NumType._('int16', 1, ffi.Int16());
  static const NumType<int> i32 = NumType._('int32', 2, ffi.Int32());
  static const NumType<int> i64 = NumType._('int64', 3, ffi.Int64());

  static const NumType<int> u8 = NumType._('uint8', 10, ffi.Uint8());
  static const NumType<int> u16 = NumType._('uint16', 11, ffi.Uint16());
  static const NumType<int> u32 = NumType._('uint32', 12, ffi.Uint32());
  static const NumType<int> u64 = NumType._('uint64', 13, ffi.Uint64());

  static const NumType<double> f32 = NumType._('float32', 22, ffi.Float());
  static const NumType<double> f64 = NumType._('float64', 23, ffi.Double());
}
