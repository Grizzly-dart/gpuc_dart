import 'dart:collection';
import 'dart:typed_data';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U16DartOnesor implements DartOnesor<int>, U16Onesor {
  @override
  Uint16List get list;

  factory U16DartOnesor(Uint16List list) => _U16DartOnesor(list);

  factory U16DartOnesor.sized(int length) => _U16DartOnesor.sized(length);

  factory U16DartOnesor.copy(Onesor<int> other) => _U16DartOnesor.copy(other);

  factory U16DartOnesor.fromList(List<int> list) =>
      _U16DartOnesor(Uint16List.fromList(list));

  @override
  U16COnesor read({Context? context}) {
    final ret = U16COnesor.sized(length, context: context);
    ret.asTypedList(length).setAll(0, this);
    return ret;
  }

  @override
  U16DartOnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U16DartOnesor(list.sublist(start, start + length));
  }

  @override
  U16DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    if (this is U16DartOnesorView) {
      start += (this as U16DartOnesorView).offset;
    }
    return U16DartOnesorView(this, start, length);
  }
}

class _U16DartOnesor
    with Onesor<int>, U16Onesor, ListMixin<int>, DartOnesor<int>, U16DartOnesor
    implements U16DartOnesor {
  @override
  final Uint16List list;

  _U16DartOnesor(this.list);

  _U16DartOnesor.sized(int length) : list = Uint16List(length);

  static _U16DartOnesor copy(Onesor<int> other) =>
      _U16DartOnesor(Uint16List.fromList(other.toList()));
}

class U16DartOnesorView
    with ListMixin<int>, Onesor<int>, U16Onesor, DartOnesor<int>, U16DartOnesor
    implements U16DartOnesor, OnesorView<int> {
  final U16DartOnesor _inner;
  @override
  final int offset;
  @override
  final int length;

  U16DartOnesorView(this._inner, this.offset, this.length);

  @override
  late final Uint16List list =
      Uint16List.sublistView(_inner.list, offset, offset + length);
}
