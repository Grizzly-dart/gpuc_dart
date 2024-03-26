import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I16COnesor implements COnesor<int>, I16Onesor {
  @override
  ffi.Pointer<ffi.Int16> get ptr;

  factory I16COnesor(ffi.Pointer<ffi.Int16> ptr, int length,
          {Context? context}) =>
      _I16COnesor(ptr, length, context: context);

  static I16COnesor copy(Onesor<int> other, {Context? context}) =>
      _I16COnesor.copy(other, context: context);

  static I16COnesor fromList(List<int> list, {Context? context}) =>
      _I16COnesor.fromList(list, context: context);

  static I16COnesor sized(int length, {Context? context}) =>
      _I16COnesor.sized(length, context: context);

  @override
  List<int> asTypedList(int length) => ptr.asTypedList(length);

  @override
  int operator [](int index) => ptr[index];

  @override
  void operator []=(int index, int value) => ptr[index] = value;

  @override
  I16COnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = I16COnesor.sized(length, context: context);
    cffi!.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  I16COnesor read({Context? context}) {
    final ret = I16COnesor.sized(length, context: context);
    cffi!.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  I16COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I16COnesorView(this, start, length);
  }
}

class _I16COnesor
    with Onesor<int>, I16Onesor, ListMixin<int>, COnesor<int>, I16COnesor
    implements I16COnesor {
  ffi.Pointer<ffi.Int16> _ptr;

  int _length;

  _I16COnesor(this._ptr, this._length, {Context? context}) {
    assert(_ptr != ffi.nullptr);
    context?.add(this);
  }

  static _I16COnesor copy(Onesor<int> other, {Context? context}) {
    final clist = _I16COnesor.sized(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _I16COnesor fromList(List<int> list, {Context? context}) {
    final ret = _I16COnesor.sized(list.length, context: context);
    ret.ptr.asTypedList(list.length).setAll(0, list);
    return ret;
  }

  static _I16COnesor sized(int length, {Context? context}) {
    final ptr = ffi.calloc<ffi.Int16>(length * Int16List.bytesPerElement);
    return _I16COnesor(ptr, length, context: context);
  }

  @override
  ffi.Pointer<ffi.Int16> get ptr => _ptr;

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
    final newPtr = cffi!.realloc(_ptr.cast(), newLength * bytesPerItem);
    if (newPtr == ffi.nullptr) {
      throw Exception('Failed to allocate memory');
    }
    _ptr = newPtr.cast();
    _length = newLength;
  }
}

class I16COnesorView
    with Onesor<int>, I16Onesor, ListMixin<int>, COnesor<int>, I16COnesor
    implements I16COnesor, COnesorView<int>, I16OnesorView {
  final I16COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  I16COnesorView(this._list, this.offset, this.length);

  @override
  late final ffi.Pointer<ffi.Int16> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }

  @override
  I16COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return I16COnesorView(_list, start + offset, length);
  }
}
