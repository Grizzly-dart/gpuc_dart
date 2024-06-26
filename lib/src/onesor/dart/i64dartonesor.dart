import 'dart:collection';
import 'dart:typed_data';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I64DartOnesor implements DartOnesor<int>, I64Onesor {
  @override
  Int64List get list;

  factory I64DartOnesor(Int64List list) => _I64DartOnesor(list);

  factory I64DartOnesor.sized(int length) => _I64DartOnesor.sized(length);

  factory I64DartOnesor.copy(Onesor<int> other) => _I64DartOnesor.copy(other);

  factory I64DartOnesor.fromList(List<int> list) =>
      _I64DartOnesor(Int64List.fromList(list));

  @override
  I64COnesor read({Context? context}) {
    final ret = I64COnesor.sized(length, context: context);
    ret.asTypedList(length).setAll(0, this);
    return ret;
  }

  @override
  I64DartOnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I64DartOnesor(list.sublist(start, start + length));
  }

  @override
  I64DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I64DartOnesorView(this, start, length);
  }
}

class _I64DartOnesor
    with
        Onesor<int>,
        I64Onesor,
        ListMixin<int>,
        DartOnesor<int>,
        DartOnesorMixin<int>,
        I64DartOnesor
    implements I64DartOnesor {
  @override
  final Int64List list;

  _I64DartOnesor(this.list);

  _I64DartOnesor.sized(int length) : list = Int64List(length);

  static _I64DartOnesor copy(Onesor<int> other) =>
      _I64DartOnesor(Int64List.fromList(other.toList()));
}

class I64DartOnesorView
    with
        ListMixin<int>,
        Onesor<int>,
        I64Onesor,
        DartOnesor<int>,
        DartOnesorViewMixin<int>,
        I64DartOnesor
    implements I64DartOnesor, DartOnesorView<int>, I64OnesorView {
  @override
  final I64DartOnesor inner;
  @override
  final int offset;
  @override
  final int length;

  I64DartOnesorView(this.inner, this.offset, this.length);

  @override
  late final Int64List list =
      Int64List.sublistView(inner.list, offset, offset + length);

  @override
  I64DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I64DartOnesorView(inner, start + offset, length);
  }
}
