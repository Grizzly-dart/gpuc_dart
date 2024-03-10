import 'dart:collection';
import 'dart:ffi' as ffi;

import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/core/core.dart';

abstract class DartList implements NList {
  factory DartList.own(List<double> list) => _DartListImpl.own(list);

  factory DartList.sized(int length, {bool growable = false}) =>
      _DartListImpl.sized(length, growable: growable);

  factory DartList.copy(NList other, {Context? context}) =>
      _DartListImpl.from(other);

  @override
  DartList slice(int start, int length, {Context? context});

  @override
  DartListView view(int start, int length);
}

class _DartListImpl extends NList
    with DartListMixin, ListMixin<double>
    implements DartList {
  final List<double> _list;

  _DartListImpl.own(this._list);

  _DartListImpl.sized(int length, {bool growable = false})
      : _list = List.filled(length, 0, growable: false);

  static _DartListImpl from(NList other) {
    return _DartListImpl.own(other.toList());
  }

  @override
  DeviceType get deviceType => DeviceType.dart;

  @override
  int get deviceId => 0;

  @override
  int get length => _list.length;

  @override
  int get lengthBytes => length * 8;

  @override
  double operator [](int index) => _list[index];

  @override
  void operator []=(int index, double value) => _list[index] = value;

  @override
  ffi.Pointer<ffi.Double> get ptr => ffi.nullptr;

  @override
  void release() {}

  @override
  set length(int newLength) => _list.length = newLength;
}

mixin DartListMixin implements DartList {
  @override
  void copyFrom(NList src) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is CList) {
      setAll(0, src.ptr.asTypedList(length));
      return;
    } else if (src is DartList) {
      setAll(0, src);
      return;
    }
    final cSrc = src.read();
    try {
      setAll(0, cSrc.ptr.asTypedList(cSrc.length));
    } finally {
      cSrc.release();
    }
  }

  @override
  void copyTo(NList dst) {
    if (lengthBytes != dst.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (dst is CList) {
      dst.ptr.asTypedList(dst.length).setAll(0, this);
      return;
    } else if (dst is DartList) {
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
  CList read({Context? context}) {
    final ret = CList.sized(length, context: context);
    ret.ptr.asTypedList(length).setAll(0, this);
    return ret;
  }

  @override
  DartList slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return DartList.own(sublist(start, start + length));
  }

  @override
  DartListView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    if (this is DartListView) {
      start += (this as DartListView)._offset;
    }
    return DartListView(this, start, length);
  }
}

class DartListView extends NList
    with DartListMixin, ListMixin<double>
    implements DartList {
  final DartList _list;
  final int _offset;
  @override
  final int length;

  DartListView(this._list, this._offset, this.length);

  @override
  double operator [](int index) => _list[_offset + index];

  @override
  void operator []=(int index, double value) => _list[_offset + index] = value;

  @override
  final int deviceId = 0;

  @override
  final DeviceType deviceType = DeviceType.dart;

  @override
  late final int lengthBytes = length * 8;

  @override
  ffi.Pointer<ffi.Double> get ptr => ffi.nullptr;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of a view');
  }
}
