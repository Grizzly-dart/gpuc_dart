import 'package:gpuc_dart/gpuc_dart.dart';

export 'f64dartonesor.dart';

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
}

abstract class DartOnesorView<T extends num>
    implements DartOnesor<T>, OnesorView<T> {}

/*
abstract class DartOnesor<T extends num> implements Onesor<T> {
  factory DartOnesor(List<T> list) => _DartOnesor(list);

  factory DartOnesor.sized(int length, {bool growable = false}) =>
      _DartOnesor.sized(length, growable: growable);

  factory DartOnesor.copy(Onesor<T> other) => _DartOnesor.copy<T>(other);

  @override
  DartOnesor<T> slice(int start, int length, {Context? context});

  @override
  DartOnesorView<T> view(int start, int length);
}




}*/
