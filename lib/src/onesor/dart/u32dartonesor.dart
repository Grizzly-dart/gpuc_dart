import 'dart:collection';
import 'dart:typed_data';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U32DartOnesor implements DartOnesor<int>, U32Onesor {
  @override
  Uint32List get list;

  factory U32DartOnesor(Uint32List list) => _U32DartOnesor(list);

  factory U32DartOnesor.sized(int length) => _U32DartOnesor.sized(length);

  factory U32DartOnesor.copy(Onesor<int> other) => _U32DartOnesor.copy(other);

  factory U32DartOnesor.fromList(List<int> list) =>
      _U32DartOnesor(Uint32List.fromList(list));

  @override
  U32COnesor read({Context? context}) {
    final ret = U32COnesor.sized(length, context: context);
    ret.asTypedList(length).setAll(0, this);
    return ret;
  }

  @override
  U32DartOnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U32DartOnesor(list.sublist(start, start + length));
  }

  @override
  U32DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    if (this is U32DartOnesorView) {
      start += (this as U32DartOnesorView).offset;
    }
    return U32DartOnesorView(this, start, length);
  }
}

class _U32DartOnesor
    with Onesor<int>, U32Onesor, ListMixin<int>, DartOnesor<int>, U32DartOnesor
    implements U32DartOnesor {
  @override
  final Uint32List list;

  _U32DartOnesor(this.list);

  _U32DartOnesor.sized(int length) : list = Uint32List(length);

  static _U32DartOnesor copy(Onesor<int> other) =>
      _U32DartOnesor(Uint32List.fromList(other.toList()));
}

class U32DartOnesorView
    with ListMixin<int>, Onesor<int>, U32Onesor, DartOnesor<int>, U32DartOnesor
    implements U32DartOnesor, OnesorView<int> {
  final U32DartOnesor _inner;
  @override
  final int offset;
  @override
  final int length;

  U32DartOnesorView(this._inner, this.offset, this.length);

  @override
  late final Uint32List list =
      Uint32List.sublistView(_inner.list, offset, offset + length);
}
