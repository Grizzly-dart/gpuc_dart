import 'dart:collection';
import 'dart:typed_data';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U64DartOnesor implements DartOnesor<int>, U64Onesor {
  @override
  Uint64List get list;

  factory U64DartOnesor(Uint64List list) => _U64DartOnesor(list);

  factory U64DartOnesor.sized(int length) => _U64DartOnesor.sized(length);

  factory U64DartOnesor.copy(Onesor<int> other) => _U64DartOnesor.copy(other);

  factory U64DartOnesor.fromList(List<int> list) =>
      _U64DartOnesor(Uint64List.fromList(list));

  @override
  U64COnesor read({Context? context}) {
    final ret = U64COnesor.sized(length, context: context);
    ret.asTypedList(length).setAll(0, this);
    return ret;
  }

  @override
  U64DartOnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U64DartOnesor(list.sublist(start, start + length));
  }

  @override
  U64DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U64DartOnesorView(this, start, length);
  }
}

class _U64DartOnesor
    with Onesor<int>, U64Onesor, ListMixin<int>, DartOnesor<int>, U64DartOnesor
    implements U64DartOnesor {
  @override
  final Uint64List list;

  _U64DartOnesor(this.list);

  _U64DartOnesor.sized(int length) : list = Uint64List(length);

  static _U64DartOnesor copy(Onesor<int> other) =>
      _U64DartOnesor(Uint64List.fromList(other.toList()));
}

class U64DartOnesorView
    with ListMixin<int>, Onesor<int>, U64Onesor, DartOnesor<int>, U64DartOnesor
    implements U64DartOnesor, DartOnesorView<int>, U64OnesorView {
  final U64DartOnesor _inner;
  @override
  final int offset;
  @override
  final int length;

  U64DartOnesorView(this._inner, this.offset, this.length);

  @override
  late final Uint64List list =
      Uint64List.sublistView(_inner.list, offset, offset + length);

  @override
  U64DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U64DartOnesorView(_inner, start + offset, length);
  }
}
