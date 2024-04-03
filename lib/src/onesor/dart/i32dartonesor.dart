import 'dart:collection';
import 'dart:typed_data';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I32DartOnesor implements DartOnesor<int>, I32Onesor {
  @override
  Int32List get list;

  factory I32DartOnesor(Int32List list) => _I32DartOnesor(list);

  factory I32DartOnesor.sized(int length) => _I32DartOnesor.sized(length);

  factory I32DartOnesor.copy(Onesor<int> other) => _I32DartOnesor.copy(other);

  factory I32DartOnesor.fromList(List<int> list) =>
      _I32DartOnesor(Int32List.fromList(list));

  @override
  I32COnesor read({Context? context}) {
    final ret = I32COnesor.sized(length, context: context);
    ret.asTypedList(length).setAll(0, this);
    return ret;
  }

  @override
  I32DartOnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I32DartOnesor(list.sublist(start, start + length));
  }

  @override
  I32DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I32DartOnesorView(this, start, length);
  }
}

class _I32DartOnesor
    with
        Onesor<int>,
        I32Onesor,
        ListMixin<int>,
        DartOnesor<int>,
        DartOnesorMixin<int>,
        I32DartOnesor
    implements I32DartOnesor {
  @override
  final Int32List list;

  _I32DartOnesor(this.list);

  _I32DartOnesor.sized(int length) : list = Int32List(length);

  static _I32DartOnesor copy(Onesor<int> other) =>
      _I32DartOnesor(Int32List.fromList(other.toList()));
}

class I32DartOnesorView
    with
        ListMixin<int>,
        Onesor<int>,
        I32Onesor,
        DartOnesor<int>,
        DartOnesorViewMixin<int>,
        I32DartOnesor
    implements I32DartOnesor, DartOnesorView<int>, I32OnesorView {
  @override
  final I32DartOnesor inner;
  @override
  final int offset;
  @override
  final int length;

  I32DartOnesorView(this.inner, this.offset, this.length);

  @override
  late final Int32List list =
      Int32List.sublistView(inner.list, offset, offset + length);

  @override
  I32DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I32DartOnesorView(inner, start + offset, length);
  }
}
