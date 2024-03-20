import 'package:gpuc_dart/gpuc_dart.dart';

export 'releaseable.dart';
export 'dart/dartonesor.dart';
export 'cuda/cuonesor.dart';
export 'c/conesor.dart';

// TODO complex onesor?
abstract class Onesor<T extends num> implements Resource, List<T> {
  DeviceType get deviceType;

  int get deviceId;

  int get lengthBytes;

  int get bytesPerItem;

  // TODO subview

  // TODO implement partial write
  void copyFrom(Onesor<T> src);

  // TODO implement partial read
  void copyTo(Onesor<T> dst);

  COnesor<T> read({Context? context});

  OnesorView<T> view(int start, int length);

  Onesor<T> slice(int start, int length, {Context? context});

  T get defaultValue;

  @override
  List<T> toList({bool growable = true}) {
    final list = List<T>.filled(length, defaultValue, growable: growable);
    copyTo(DartOnesor<T>(list));
    return list;
  }

  @override
  void release();
}

abstract class OnesorView<T extends num> extends Onesor<T> {
  int get offset;
}

extension OnesorExtension<T extends num> on Onesor<T> {
  Device get device => Device(deviceType, deviceId);
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
