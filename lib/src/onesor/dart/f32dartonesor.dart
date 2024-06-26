import 'dart:collection';
import 'dart:typed_data';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class F32DartOnesor implements DartOnesor<double>, F32Onesor {
  @override
  Float32List get list;

  factory F32DartOnesor(Float32List list) => _F32DartOnesor(list);

  factory F32DartOnesor.sized(int length) => _F32DartOnesor.sized(length);

  factory F32DartOnesor.copy(Onesor<double> other) =>
      _F32DartOnesor.copy(other);

  factory F32DartOnesor.fromList(List<double> list) =>
      _F32DartOnesor(Float32List.fromList(list));

  @override
  F32COnesor read({Context? context}) {
    final ret = F32COnesor.sized(length, context: context);
    ret.asTypedList(length).setAll(0, this);
    return ret;
  }

  @override
  F32DartOnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return F32DartOnesor(list.sublist(start, start + length));
  }

  @override
  F32DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return F32DartOnesorView(this, start, length);
  }
}

class _F32DartOnesor
    with
        Onesor<double>,
        F32Onesor,
        ListMixin<double>,
        DartOnesor<double>,
        DartOnesorMixin<double>,
        F32DartOnesor
    implements F32DartOnesor {
  @override
  final Float32List list;

  _F32DartOnesor(this.list);

  _F32DartOnesor.sized(int length) : list = Float32List(length);

  static _F32DartOnesor copy(Onesor<double> other) =>
      _F32DartOnesor(Float32List.fromList(other.toList()));
}

class F32DartOnesorView
    with
        ListMixin<double>,
        Onesor<double>,
        F32Onesor,
        DartOnesor<double>,
        DartOnesorViewMixin<double>,
        F32DartOnesor
    implements F32DartOnesor, DartOnesorView<double>, F32OnesorView {
  @override
  final F32DartOnesor inner;
  @override
  final int offset;
  @override
  final int length;

  F32DartOnesorView(this.inner, this.offset, this.length);

  @override
  late final Float32List list =
      Float32List.sublistView(inner.list, offset, offset + length);

  @override
  F32DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return F32DartOnesorView(inner, start + offset, length);
  }
}
