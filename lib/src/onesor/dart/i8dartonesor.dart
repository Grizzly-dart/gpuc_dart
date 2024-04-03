import 'dart:collection';
import 'dart:typed_data';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I8DartOnesor implements DartOnesor<int>, I8Onesor {
  @override
  Int8List get list;

  factory I8DartOnesor(Int8List list) => _I8DartOnesor(list);

  factory I8DartOnesor.sized(int length) => _I8DartOnesor.sized(length);

  factory I8DartOnesor.copy(Onesor<int> other) => _I8DartOnesor.copy(other);

  factory I8DartOnesor.fromList(List<int> list) =>
      _I8DartOnesor(Int8List.fromList(list));

  @override
  I8COnesor read({Context? context}) {
    final ret = I8COnesor.sized(length, context: context);
    ret.asTypedList(length).setAll(0, this);
    return ret;
  }

  @override
  I8DartOnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I8DartOnesor(list.sublist(start, start + length));
  }

  @override
  I8DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I8DartOnesorView(this, start, length);
  }
}

class _I8DartOnesor
    with
        Onesor<int>,
        I8Onesor,
        ListMixin<int>,
        DartOnesor<int>,
        DartOnesorMixin<int>,
        I8DartOnesor
    implements I8DartOnesor {
  @override
  final Int8List list;

  _I8DartOnesor(this.list);

  _I8DartOnesor.sized(int length) : list = Int8List(length);

  static _I8DartOnesor copy(Onesor<int> other) =>
      _I8DartOnesor(Int8List.fromList(other.toList()));
}

class I8DartOnesorView
    with
        ListMixin<int>,
        Onesor<int>,
        I8Onesor,
        DartOnesor<int>,
        DartOnesorViewMixin<int>,
        I8DartOnesor
    implements I8DartOnesor, DartOnesorView<int>, I8OnesorView {
  @override
  final I8DartOnesor inner;
  @override
  final int offset;
  @override
  final int length;

  I8DartOnesorView(this.inner, this.offset, this.length);

  @override
  late final Int8List list =
      Int8List.sublistView(inner.list, offset, offset + length);

  @override
  I8DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I8DartOnesorView(inner, start + offset, length);
  }
}
