import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:gpuc_dart/src/core/c.dart';
import 'package:gpuc_dart/src/core/cuda.dart';
import 'package:gpuc_dart/src/core/releaseable.dart';

class DartList extends NList {
  final List<double> list;

  DartList.fromList(this.list);

  static DartList copy(NList other) {
    if (other is DartList) {
      final list = Float64List.fromList(other.list);
      return DartList.fromList(list);
    } else if (other is CList) {
      final list = other.ptr.asTypedList(other.length);
      return DartList.fromList(list);
    }
    final cSrc = other.read();
    try {
      final list = cSrc.ptr.asTypedList(cSrc.length);
      return DartList.fromList(list);
    } finally {
      cSrc.release();
    }
  }

  @override
  DeviceType get deviceType => DeviceType.dart;

  @override
  int get deviceId => 0;

  @override
  int get length => list.length;

  @override
  int get lengthBytes => length * 8;

  @override
  double operator [](int index) => list[index];

  @override
  void operator []=(int index, double value) {
    list[index] = value;
  }

  @override
  ffi.Pointer<ffi.Double> get ptr => ffi.nullptr;

  @override
  void release() {}

  @override
  void copyFrom(NList src) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is CList) {
      list.setAll(0, src.ptr.asTypedList(length));
      return;
    } else if (src is DartList) {
      list.setAll(0, src.list);
      return;
    }
    final cSrc = src.read();
    try {
      list.setAll(0, cSrc.ptr.asTypedList(cSrc.length));
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
      dst.ptr.asTypedList(dst.length).setAll(0, list);
      return;
    } else if (dst is DartList) {
      dst.list.setAll(0, list);
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
    final clist = CList.allocate(list.length, context: context);
    clist.ptr.asTypedList(list.length).setAll(0, list);
    return clist;
  }
}