import 'dart:collection';
import 'dart:ffi' as ffi;

import 'package:gpuc_dart/gpuc_dart.dart';

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

class _DartOnesor<T extends num>
    with DartOnesorMixin<T>, ListMixin<T>
    implements DartOnesor<T> {
  final List<T> _list;

  _DartOnesor(this._list);

  _DartOnesor.sized(int length, {bool growable = false})
      : _list = List.filled(length, (T == int ? 0 : 0.0) as T, growable: false);

  static _DartOnesor<T> copy<T extends num>(Onesor<T> other) =>
      _DartOnesor(other.toList());

  @override
  DeviceType get deviceType => DeviceType.dart;

  @override
  int get deviceId => 0;

  @override
  int get length => _list.length;

  @override
  int get lengthBytes => length * 8;

  @override
  T operator [](int index) => _list[index];

  @override
  void operator []=(int index, T value) => _list[index] = value;

  @override
  void release() {}

  @override
  set length(int newLength) => _list.length = newLength;
}

class DartOnesorView<T extends num>
    with DartOnesorMixin<T>, ListMixin<T>
    implements DartOnesor<T>, OnesorView<T> {
  final DartOnesor<T> _list;
  @override
  final int offset;
  @override
  final int length;

  DartOnesorView(this._list, this.offset, this.length);

  @override
  T operator [](int index) => _list[offset + index];

  @override
  void operator []=(int index, T value) => _list[offset + index] = value;

  @override
  final int deviceId = 0;

  @override
  final DeviceType deviceType = DeviceType.dart;

  @override
  late final int lengthBytes = length * 8;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of a view');
  }
}

mixin DartOnesorMixin<T extends num> implements Onesor<T> {
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
  COnesor<T> read({Context? context}) {
    final ret = COnesor<T>.sized(length, context: context);
    ret.asTypedList(length).setAll(0, this);
    return ret;
  }

  @override
  DartOnesor<T> slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return DartOnesor<T>(sublist(start, start + length));
  }

  @override
  DartOnesorView<T> view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    if (this is DartOnesorView<T>) {
      start += (this as DartOnesorView<T>).offset;
    }
    return DartOnesorView(this as DartOnesor<T>, start, length);
  }

  @override
  T get defaultValue {
    if (T == int) {
      return 0 as T;
    } else if (T == double) {
      return 0.0 as T;
    }
    throw UnsupportedError('Unsupported type');
  }

  @override
  int get bytesPerItem {
    if (T == int) {
      return 4;
    } else if (T == double) {
      return 8;
    }
    throw UnsupportedError('Unsupported type');
  }
}
