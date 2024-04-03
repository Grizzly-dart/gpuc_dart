part of 'conesor.dart';

abstract mixin class I32COnesor implements COnesor<int>, I32Onesor {
  @override
  ffi.Pointer<ffi.Int32> get ptr;

  static I32COnesor copy(Onesor<int> other, {Context? context}) =>
      _I32COnesor.copy(other, context: context);

  static I32COnesor fromList(List<int> list, {Context? context}) =>
      _I32COnesor.fromList(list, context: context);

  static I32COnesor sized(int length, {Context? context}) =>
      _I32COnesor.sized(length, context: context);

  @override
  List<int> asTypedList(int length) => ptr.asTypedList(length);

  @override
  int operator [](int index) => ptr[index];

  @override
  void operator []=(int index, int value) => ptr[index] = value;

  @override
  I32COnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = I32COnesor.sized(length, context: context);
    cffi!.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  I32COnesor read({Context? context}) {
    final ret = I32COnesor.sized(length, context: context);
    cffi!.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  I32COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I32COnesorView(this, length, start);
  }
}

class _I32COnesor
    with
        Onesor<int>,
        I32Onesor,
        ListMixin<int>,
        _COnesorMixin<int>,
        COnesor<int>,
        I32COnesor
    implements I32COnesor {
  @override
  final CPtr<ffi.Int32> _ptr;

  int _length;

  _I32COnesor(this._ptr, this._length, {Context? context}) {
    context?.add(this);
  }

  static _I32COnesor copy(Onesor<int> other, {Context? context}) {
    final clist = _I32COnesor.sized(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _I32COnesor fromList(List<int> list, {Context? context}) {
    final ret = _I32COnesor.sized(list.length, context: context);
    ret.ptr.asTypedList(list.length).setAll(0, list);
    return ret;
  }

  static _I32COnesor sized(int length, {Context? context}) =>
      _I32COnesor(CPtr.allocate(i32.bytes, count: length), length,
          context: context);

  @override
  ffi.Pointer<ffi.Int32> get ptr => _ptr.ptr;

  @override
  int get length => _length;
}

class I32COnesorView
    with
        Onesor<int>,
        OnesorView<int>,
        I32Onesor,
        ListMixin<int>,
        COnesor<int>,
        _COnesorViewMixin<int>,
        I32COnesor
    implements I32COnesor, COnesorView<int>, I32OnesorView {
  @override
  final I32COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  I32COnesorView(this._list, this.length, this.offset);

  @override
  late final ffi.Pointer<ffi.Int32> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }

  @override
  I32COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I32COnesorView(_list, length, start + offset);
  }
}
