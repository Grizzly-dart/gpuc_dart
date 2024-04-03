import 'dart:collection';
import 'dart:typed_data';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I16DartOnesor implements DartOnesor<int>, I16Onesor {
  @override
  Int16List get list;

  factory I16DartOnesor(Int16List list) => _I16DartOnesor(list);

  factory I16DartOnesor.sized(int length) => _I16DartOnesor.sized(length);

  factory I16DartOnesor.copy(Onesor<int> other) =>
      _I16DartOnesor.copy(other);

  factory I16DartOnesor.fromList(List<int> list) =>
      _I16DartOnesor(Int16List.fromList(list));

  @override
  I16COnesor read({Context? context}) {
    final ret = I16COnesor.sized(length, context: context);
    ret.asTypedList(length).setAll(0, this);
    return ret;
  }

  @override
  I16DartOnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I16DartOnesor(list.sublist(start, start + length));
  }

  @override
  I16DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I16DartOnesorView(this, start, length);
  }
}

class _I16DartOnesor
    with
        Onesor<int>,
        I16Onesor,
        ListMixin<int>,
        DartOnesor<int>,
        DartOnesorMixin<int>,
        I16DartOnesor
    implements I16DartOnesor {
  @override
  final Int16List list;

  _I16DartOnesor(this.list);

  _I16DartOnesor.sized(int length) : list = Int16List(length);

  static _I16DartOnesor copy(Onesor<int> other) =>
      _I16DartOnesor(Int16List.fromList(other.toList()));
}

class I16DartOnesorView
    with
        ListMixin<int>,
        Onesor<int>,
        I16Onesor,
        DartOnesor<int>,
        DartOnesorViewMixin<int>,
        I16DartOnesor
    implements I16DartOnesor, DartOnesorView<int>, I16OnesorView {
  @override
  final I16DartOnesor inner;
  @override
  final int offset;
  @override
  final int length;

  I16DartOnesorView(this.inner, this.offset, this.length);

  @override
  late final Int16List list =
  Int16List.sublistView(inner.list, offset, offset + length);

  @override
  I16DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I16DartOnesorView(inner, start + offset, length);
  }
}
