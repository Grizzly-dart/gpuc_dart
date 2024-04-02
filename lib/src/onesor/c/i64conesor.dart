part of 'conesor.dart';

abstract mixin class I64COnesor implements COnesor<int>, I64Onesor {
  @override
  ffi.Pointer<ffi.Int64> get ptr;

  static I64COnesor copy(Onesor<int> other, {Context? context}) =>
      _I64COnesor.copy(other, context: context);

  static I64COnesor fromList(List<int> list, {Context? context}) =>
      _I64COnesor.fromList(list, context: context);

  static I64COnesor sized(int length, {Context? context}) =>
      _I64COnesor.sized(length, context: context);

  @override
  List<int> asTypedList(int length) => ptr.asTypedList(length);

  @override
  int operator [](int index) => ptr[index];

  @override
  void operator []=(int index, int value) => ptr[index] = value;

  @override
  I64COnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = I64COnesor.sized(length, context: context);
    cffi!.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  I64COnesor read({Context? context}) {
    final ret = I64COnesor.sized(length, context: context);
    cffi!.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  I64COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I64COnesorView(this, length, start);
  }
}

class _I64COnesor
    with
        Onesor<int>,
        ListMixin<int>,
        _COnesorMixin<int>,
        COnesor<int>,
        I64Onesor,
        I64COnesor
    implements I64Onesor, I64COnesor {
  @override
  final CPtr<ffi.Int64> _ptr;

  int _length;

  _I64COnesor(this._ptr, this._length, {Context? context}) {
    context?.add(this);
  }

  static _I64COnesor copy(Onesor<int> other, {Context? context}) {
    final clist = _I64COnesor.sized(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _I64COnesor fromList(List<int> list, {Context? context}) {
    final ret = _I64COnesor.sized(list.length, context: context);
    ret.ptr.asTypedList(list.length).setAll(0, list);
    return ret;
  }

  static _I64COnesor sized(int length, {Context? context}) =>
      _I64COnesor(CPtr.allocate(i64.bytes, count: length), length,
          context: context);

  @override
  ffi.Pointer<ffi.Int64> get ptr => _ptr.ptr;

  @override
  int get length => _length;
}

class I64COnesorView
    with Onesor<int>, I64Onesor, ListMixin<int>, COnesor<int>, I64COnesor
    implements I64COnesor, COnesorView<int>, I64OnesorView {
  final I64COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  I64COnesorView(this._list, this.length, this.offset);

  @override
  late final ffi.Pointer<ffi.Int64> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }

  @override
  I64COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I64COnesorView(_list, length, start + offset);
  }
}
