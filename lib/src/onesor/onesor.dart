import 'dart:ffi' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

export 'releaseable.dart';
export 'dart/dartonesor.dart';
export 'cuda/cuonesor.dart';
export 'c/conesor.dart';

// TODO complex onesor?
abstract mixin class Onesor<T extends num>
    implements Resource, List<T> {
  DeviceType get deviceType;

  int get deviceId;

  NumType<T> get type;

  int get lengthBytes => length * bytesPerItem;

  int get bytesPerItem => type.bytes;

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
