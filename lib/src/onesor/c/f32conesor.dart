import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class F32COnesor implements COnesor<double>, F32Onesor {
  @override
  ffi.Pointer<ffi.Float> get ptr;

  factory F32COnesor(ffi.Pointer<ffi.Float> ptr, int length,
          {Context? context}) =>
      _F32COnesor(ptr, length, context: context);

  static F32COnesor copy(Onesor<double> other, {Context? context}) =>
      _F32COnesor.copy(other, context: context);

  static F32COnesor fromList(List<double> list, {Context? context}) =>
      _F32COnesor.fromList(list, context: context);

  static F32COnesor sized(int length, {Context? context}) =>
      _F32COnesor.sized(length, context: context);

  @override
  List<double> asTypedList(int length) => ptr.asTypedList(length);

  @override
  double operator [](int index) => ptr[index];

  @override
  void operator []=(int index, double value) => ptr[index] = value;

  @override
  F32COnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = F32COnesor.sized(length, context: context);
    CFFI.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  F32COnesor read({Context? context}) {
    final ret = F32COnesor.sized(length, context: context);
    CFFI.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  F32COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    if (this is COnesorView<double>) {
      start += (this as COnesorView<double>).offset;
    }
    return F32COnesorView(this, start, length);
  }
}

class _F32COnesor
    with
        Onesor<double>,
        F32Onesor,
        ListMixin<double>,
        COnesor<double>,
        F32COnesor
    implements F32COnesor {
  ffi.Pointer<ffi.Float> _ptr;

  int _length;

  _F32COnesor(this._ptr, this._length, {Context? context}) {
    assert(_ptr != ffi.nullptr);
    context?.add(this);
  }

  static _F32COnesor copy(Onesor<double> other, {Context? context}) {
    final clist = _F32COnesor.sized(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _F32COnesor fromList(List<double> list, {Context? context}) {
    final ret = _F32COnesor.sized(list.length, context: context);
    ret.ptr.asTypedList(list.length).setAll(0, list);
    return ret;
  }

  static _F32COnesor sized(int length, {Context? context}) {
    final ptr = ffi.calloc<ffi.Float>(length * Float64List.bytesPerElement);
    return _F32COnesor(ptr, length, context: context);
  }

  @override
  ffi.Pointer<ffi.Float> get ptr => _ptr;

  @override
  int get length => _length;

  @override
  void release() {
    if (_ptr == ffi.nullptr) return;
    ffi.malloc.free(_ptr);
    _ptr = ffi.nullptr;
  }

  @override
  set length(int newLength) {
    final newPtr = CFFI.realloc(_ptr.cast(), newLength * bytesPerItem);
    if (newPtr == ffi.nullptr) {
      throw Exception('Failed to allocate memory');
    }
    _ptr = newPtr.cast();
    _length = newLength;
  }
}

class F32COnesorView
    with
        Onesor<double>,
        F32Onesor,
        ListMixin<double>,
        COnesor<double>,
        F32COnesor
    implements F32COnesor, COnesorView<double> {
  final COnesor<double> _list;

  @override
  final int offset;

  @override
  final int length;

  F32COnesorView(this._list, this.offset, this.length);

  @override
  late final ffi.Pointer<ffi.Float> ptr =
      _list.ptr.cast<ffi.Float>() + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }
}
