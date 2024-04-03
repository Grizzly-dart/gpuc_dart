part of 'conesor.dart';

abstract mixin class U16COnesor implements COnesor<int>, U16Onesor {
  @override
  ffi.Pointer<ffi.Uint16> get ptr;

  static U16COnesor copy(Onesor<int> other, {Context? context}) =>
      _U16COnesor.copy(other, context: context);

  static U16COnesor fromList(List<int> list, {Context? context}) =>
      _U16COnesor.fromList(list, context: context);

  static U16COnesor sized(int length, {Context? context}) =>
      _U16COnesor.sized(length, context: context);

  @override
  List<int> asTypedList(int length) => ptr.asTypedList(length);

  @override
  int operator [](int index) => ptr[index];

  @override
  void operator []=(int index, int value) => ptr[index] = value;

  @override
  U16COnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = U16COnesor.sized(length, context: context);
    tc.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  U16COnesor read({Context? context}) {
    final ret = U16COnesor.sized(length, context: context);
    tc.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  U16COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U16COnesorView(this, length, start);
  }
}

class _U16COnesor
    with
        Onesor<int>,
        U16Onesor,
        ListMixin<int>,
        _COnesorMixin<int>,
        COnesor<int>,
        U16COnesor
    implements U16COnesor {
  @override
  final CPtr<ffi.Uint16> _ptr;

  int _length;

  _U16COnesor(this._ptr, this._length, {Context? context}) {
    context?.add(this);
  }

  static _U16COnesor copy(Onesor<int> other, {Context? context}) {
    final clist = _U16COnesor.sized(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _U16COnesor fromList(List<int> list, {Context? context}) {
    final ret = _U16COnesor.sized(list.length, context: context);
    ret.ptr.asTypedList(list.length).setAll(0, list);
    return ret;
  }

  static _U16COnesor sized(int length, {Context? context}) =>
      _U16COnesor(CPtr.allocate(u16.bytes, count: length), length,
          context: context);

  @override
  ffi.Pointer<ffi.Uint16> get ptr => _ptr.ptr;

  @override
  int get length => _length;
}

class U16COnesorView
    with
        Onesor<int>,
        OnesorView<int>,
        U16Onesor,
        ListMixin<int>,
        COnesor<int>,
        _COnesorViewMixin<int>,
        U16COnesor
    implements U16COnesor, COnesorView<int>, U16OnesorView {
  @override
  final U16COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  U16COnesorView(this._list, this.length, this.offset);

  @override
  late final ffi.Pointer<ffi.Uint16> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }

  @override
  U16COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U16COnesorView(_list, length, start + offset);
  }
}
