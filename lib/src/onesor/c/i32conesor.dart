import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I32COnesor implements COnesor<int>, I32Onesor {
  @override
  ffi.Pointer<ffi.Int32> get ptr;

  factory I32COnesor(ffi.Pointer<ffi.Int32> ptr, int length,
          {Context? context}) =>
      _I32COnesor(ptr, length, context: context);

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
    CFFI.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  I32COnesor read({Context? context}) {
    final ret = I32COnesor.sized(length, context: context);
    CFFI.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  I32COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    if (this is COnesorView<int>) {
      start += (this as COnesorView<int>).offset;
    }
    return I32COnesorView(this, start, length);
  }
}

class _I32COnesor
    with Onesor<int>, I32Onesor, ListMixin<int>, COnesor<int>, I32COnesor
    implements I32COnesor {
  ffi.Pointer<ffi.Int32> _ptr;

  int _length;

  _I32COnesor(this._ptr, this._length, {Context? context}) {
    assert(_ptr != ffi.nullptr);
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

  static _I32COnesor sized(int length, {Context? context}) {
    final ptr = ffi.calloc<ffi.Int32>(length * Int32List.bytesPerElement);
    return _I32COnesor(ptr, length, context: context);
  }

  @override
  ffi.Pointer<ffi.Int32> get ptr => _ptr;

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

class I32COnesorView
    with Onesor<int>, I32Onesor, ListMixin<int>, COnesor<int>, I32COnesor
    implements I32COnesor, COnesorView<int> {
  final I32COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  I32COnesorView(this._list, this.offset, this.length);

  @override
  late final ffi.Pointer<ffi.Int32> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }
}
