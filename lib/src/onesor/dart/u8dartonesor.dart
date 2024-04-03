import 'dart:collection';
import 'dart:typed_data';

import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U8DartOnesor implements DartOnesor<int>, U8Onesor {
  @override
  Uint8List get list;

  factory U8DartOnesor(Uint8List list) => _U8DartOnesor(list);

  factory U8DartOnesor.sized(int length) => _U8DartOnesor.sized(length);

  factory U8DartOnesor.copy(Onesor<int> other) => _U8DartOnesor.copy(other);

  factory U8DartOnesor.fromList(List<int> list) =>
      _U8DartOnesor(Uint8List.fromList(list));

  @override
  U8COnesor read({Context? context}) {
    final ret = U8COnesor.sized(length, context: context);
    ret.asTypedList(length).setAll(0, this);
    return ret;
  }

  @override
  U8DartOnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U8DartOnesor(list.sublist(start, start + length));
  }

  @override
  U8DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U8DartOnesorView(this, start, length);
  }
}

class _U8DartOnesor
    with
        Onesor<int>,
        U8Onesor,
        ListMixin<int>,
        DartOnesor<int>,
        DartOnesorMixin<int>,
        U8DartOnesor
    implements U8DartOnesor {
  @override
  final Uint8List list;

  _U8DartOnesor(this.list);

  _U8DartOnesor.sized(int length) : list = Uint8List(length);

  static _U8DartOnesor copy(Onesor<int> other) =>
      _U8DartOnesor(Uint8List.fromList(other.toList()));
}

class U8DartOnesorView
    with
        ListMixin<int>,
        Onesor<int>,
        U8Onesor,
        DartOnesor<int>,
        DartOnesorViewMixin<int>,
        U8DartOnesor
    implements U8DartOnesor, DartOnesorView<int>, U8OnesorView {
  @override
  final U8DartOnesor inner;
  @override
  final int offset;
  @override
  final int length;

  U8DartOnesorView(this.inner, this.offset, this.length);

  @override
  late final Uint8List list =
      Uint8List.sublistView(inner.list, offset, offset + length);

  @override
  U8DartOnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U8DartOnesorView(inner, start + offset, length);
  }
}
