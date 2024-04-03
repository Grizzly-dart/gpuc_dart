part of 'conesor.dart';

abstract mixin class F64COnesor implements COnesor<double>, F64Onesor {
  @override
  ffi.Pointer<ffi.Double> get ptr;

  static F64COnesor copy(Onesor<double> other, {Context? context}) =>
      _F64COnesor.copy(other, context: context);

  static F64COnesor fromList(List<double> list, {Context? context}) =>
      _F64COnesor.fromList(list, context: context);

  static F64COnesor sized(int length, {Context? context}) =>
      _F64COnesor.sized(length, context: context);

  @override
  List<double> asTypedList(int length) => ptr.asTypedList(length);

  @override
  double operator [](int index) => ptr[index];

  @override
  void operator []=(int index, double value) => ptr[index] = value;

  @override
  F64COnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = F64COnesor.sized(length, context: context);
    tc.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  F64COnesor read({Context? context}) {
    final ret = F64COnesor.sized(length, context: context);
    tc.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  F64COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return F64COnesorView(this, length, start);
  }
}

class _F64COnesor
    with
        Onesor<double>,
        F64Onesor,
        ListMixin<double>,
        _COnesorMixin<double>,
        COnesor<double>,
        F64COnesor
    implements F64COnesor {
  @override
  final CPtr<ffi.Double> _ptr;

  int _length;

  _F64COnesor(this._ptr, this._length, {Context? context}) {
    context?.add(this);
  }

  static _F64COnesor copy(Onesor<double> other, {Context? context}) {
    final clist = _F64COnesor.sized(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _F64COnesor fromList(List<double> list, {Context? context}) {
    final ret = _F64COnesor.sized(list.length, context: context);
    ret.ptr.asTypedList(list.length).setAll(0, list);
    return ret;
  }

  static _F64COnesor sized(int length, {Context? context}) =>
      _F64COnesor(CPtr.allocate(f64.bytes, count: length), length,
          context: context);

  @override
  ffi.Pointer<ffi.Double> get ptr => _ptr.ptr;

  @override
  int get length => _length;
}

class F64COnesorView
    with
        Onesor<double>,
        OnesorView<double>,
        F64Onesor,
        ListMixin<double>,
        COnesor<double>,
        _COnesorViewMixin<double>,
        F64COnesor
    implements F64COnesor, COnesorView<double>, F64OnesorView {
  @override
  final F64COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  F64COnesorView(this._list, this.length, this.offset);

  @override
  late final ffi.Pointer<ffi.Double> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }

  @override
  F64COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return F64COnesorView(_list, length, offset + start);
  }
}
