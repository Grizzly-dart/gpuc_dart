import 'dart:io';

import 'package:gpuc_dart/gpuc_dart.dart';

export 'f64dartonesor.dart';
export 'f32dartonesor.dart';
export 'u64dartonesor.dart';
export 'i64dartonesor.dart';
export 'i32dartonesor.dart';
export 'u32dartonesor.dart';
export 'i16dartonesor.dart';
export 'u16dartonesor.dart';
export 'i8dartonesor.dart';
export 'u8dartonesor.dart';

abstract mixin class DartOnesor<T extends num> implements Onesor<T> {
  List<T> get list;

  @override
  DeviceType get deviceType => DeviceType.dart;

  @override
  int get deviceId => 0;

  @override
  int get length => list.length;

  @override
  T operator [](int index) => list[index];

  @override
  void operator []=(int index, T value) => list[index] = value;

  @override
  void copyFrom(Onesor<T> src) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is COnesor<T>) {
      setAll(0, src.asTypedList(length));
      return;
    } else if (src is DartOnesor<T>) {
      setAll(0, src);
      return;
    }
    final cSrc = src.read();
    try {
      setAll(0, cSrc.asTypedList(cSrc.length));
    } finally {
      cSrc.release();
    }
  }

  @override
  void copyTo(Onesor<T> dst) {
    if (lengthBytes != dst.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (dst is COnesor<T>) {
      dst.asTypedList(dst.length).setAll(0, this);
      return;
    } else if (dst is DartOnesor<T>) {
      dst.setAll(0, this);
      return;
    }
    final cSrc = read();
    try {
      dst.copyFrom(cSrc);
    } finally {
      cSrc.release();
    }
  }

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length');
  }

  final Finalizer finalizer = Finalizer((other) {
    if (other is Resource) {
      other.release();
    } else {
      stdout.writeln('Cannot release $other');
    }
  });
}

mixin DartOnesorMixin<T extends num> implements DartOnesor<T> {
  @override
  void coRelease(Resource other) {
    finalizer.attach(list, other, detach: other);
  }

  @override
  void detachCoRelease(Resource other) {
    finalizer.detach(other);
  }
}

abstract class DartOnesorView<T extends num>
    implements DartOnesor<T>, OnesorView<T> {}

mixin DartOnesorViewMixin<T extends num> implements DartOnesorView<T> {
  DartOnesor<T> get inner;

  @override
  void coRelease(Resource other) {
    inner.coRelease(other);
  }

  @override
  void detachCoRelease(Resource other) {
    inner.detachCoRelease(other);
  }
}
