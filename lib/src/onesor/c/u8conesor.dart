part of 'conesor.dart';

abstract mixin class U8COnesor implements COnesor<int>, U8Onesor {
  @override
  ffi.Pointer<ffi.Uint8> get ptr;

  static U8COnesor copy(Onesor<int> other, {Context? context}) =>
      _U8COnesor.copy(other, context: context);

  static U8COnesor fromList(List<int> list, {Context? context}) =>
      _U8COnesor.fromList(list, context: context);

  static U8COnesor sized(int length, {Context? context}) =>
      _U8COnesor.sized(length, context: context);

  @override
  List<int> asTypedList(int length) => ptr.asTypedList(length);

  @override
  int operator [](int index) => ptr[index];

  @override
  void operator []=(int index, int value) => ptr[index] = value;

  @override
  U8COnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = U8COnesor.sized(length, context: context);
    tc.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  U8COnesor read({Context? context}) {
    final ret = U8COnesor.sized(length, context: context);
    tc.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  U8COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U8COnesorView(this, length, start);
  }
}

class _U8COnesor
    with
        Onesor<int>,
        U8Onesor,
        ListMixin<int>,
        _COnesorMixin<int>,
        COnesor<int>,
        U8COnesor
    implements U8COnesor {
  @override
  final CPtr<ffi.Uint8> _ptr;

  int _length;

  _U8COnesor(this._ptr, this._length, {Context? context}) {
    context?.add(this);
  }

  static _U8COnesor copy(Onesor<int> other, {Context? context}) {
    final clist = _U8COnesor.sized(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _U8COnesor fromList(List<int> list, {Context? context}) {
    final ret = _U8COnesor.sized(list.length, context: context);
    ret.ptr.asTypedList(list.length).setAll(0, list);
    return ret;
  }

  static _U8COnesor sized(int length, {Context? context}) =>
      _U8COnesor(CPtr.allocate(u8.bytes, count: length), length,
          context: context);

  @override
  ffi.Pointer<ffi.Uint8> get ptr => _ptr.ptr;

  @override
  int get length => _length;
}

class U8COnesorView
    with
        Onesor<int>,
        OnesorView<int>,
        U8Onesor,
        ListMixin<int>,
        COnesor<int>,
        _COnesorViewMixin<int>,
        U8COnesor
    implements U8COnesor, COnesorView<int>, U8OnesorView {
  @override
  final U8COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  U8COnesorView(this._list, this.length, this.offset);

  @override
  late final ffi.Pointer<ffi.Uint8> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }

  @override
  U8COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U8COnesorView(_list, length, start + offset);
  }
}
