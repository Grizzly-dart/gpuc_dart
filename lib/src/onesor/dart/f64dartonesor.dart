import 'dart:collection';
import 'dart:typed_data';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class F64DartOnesor implements DartOnesor<double>, F64Onesor {
  @override
  Float64List get list;

  factory F64DartOnesor(Float64List list) => _F64DartOnesor(list);

  factory F64DartOnesor.sized(int length) => _F64DartOnesor.sized(length);

  factory F64DartOnesor.copy(Onesor<double> other) =>
      _F64DartOnesor.copy(other);

  factory F64DartOnesor.fromList(List<double> list) =>
      _F64DartOnesor(Float64List.fromList(list));

  @override
  F64COnesor read({Context? context}) {
    final ret = F64COnesor.sized(length, context: context);
    ret.asTypedList(length).setAll(0, this);
    return ret;
  }

  @override
  F64DartOnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return F64DartOnesor(list.sublist(start, start + length));
  }

  @override
  F64DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return F64DartOnesorView(this, start, length);
  }
}

class _F64DartOnesor
    with
        Onesor<double>,
        F64Onesor,
        ListMixin<double>,
        DartOnesor<double>,
        F64DartOnesor
    implements F64DartOnesor {
  @override
  final Float64List list;

  _F64DartOnesor(this.list);

  _F64DartOnesor.sized(int length) : list = Float64List(length);

  static _F64DartOnesor copy(Onesor<double> other) =>
      _F64DartOnesor(Float64List.fromList(other.toList()));
}

class F64DartOnesorView
    with
        ListMixin<double>,
        Onesor<double>,
        F64Onesor,
        DartOnesor<double>,
        F64DartOnesor
    implements F64DartOnesor, DartOnesorView<double>, F64OnesorView {
  final F64DartOnesor _inner;
  @override
  final int offset;
  @override
  final int length;

  F64DartOnesorView(this._inner, this.offset, this.length);

  @override
  late final Float64List list =
      Float64List.sublistView(_inner.list, offset, offset + length);

  @override
  F64DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return F64DartOnesorView(_inner, start + offset, length);
  }
}
