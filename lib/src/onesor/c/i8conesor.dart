import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I8COnesor implements COnesor<int>, I8Onesor {
  @override
  ffi.Pointer<ffi.Int8> get ptr;

  factory I8COnesor(ffi.Pointer<ffi.Int8> ptr, int length,
          {Context? context}) =>
      _I8COnesor(ptr, length, context: context);

  static I8COnesor copy(Onesor<int> other, {Context? context}) =>
      _I8COnesor.copy(other, context: context);

  static I8COnesor fromList(List<int> list, {Context? context}) =>
      _I8COnesor.fromList(list, context: context);

  static I8COnesor sized(int length, {Context? context}) =>
      _I8COnesor.sized(length, context: context);

  @override
  List<int> asTypedList(int length) => ptr.asTypedList(length);

  @override
  int operator [](int index) => ptr[index];

  @override
  void operator []=(int index, int value) => ptr[index] = value;

  @override
  I8COnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = I8COnesor.sized(length, context: context);
    CFFI.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  I8COnesor read({Context? context}) {
    final ret = I8COnesor.sized(length, context: context);
    CFFI.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  I8COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    if (this is COnesorView<int>) {
      start += (this as COnesorView<int>).offset;
    }
    return I8COnesorView(this, start, length);
  }
}

class _I8COnesor
    with Onesor<int>, I8Onesor, ListMixin<int>, COnesor<int>, I8COnesor
    implements I8COnesor {
  ffi.Pointer<ffi.Int8> _ptr;

  int _length;

  _I8COnesor(this._ptr, this._length, {Context? context}) {
    assert(_ptr != ffi.nullptr);
    context?.add(this);
  }

  static _I8COnesor copy(Onesor<int> other, {Context? context}) {
    final clist = _I8COnesor.sized(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _I8COnesor fromList(List<int> list, {Context? context}) {
    final ret = _I8COnesor.sized(list.length, context: context);
    ret.ptr.asTypedList(list.length).setAll(0, list);
    return ret;
  }

  static _I8COnesor sized(int length, {Context? context}) {
    final ptr = ffi.calloc<ffi.Int8>(length * Int8List.bytesPerElement);
    return _I8COnesor(ptr, length, context: context);
  }

  @override
  ffi.Pointer<ffi.Int8> get ptr => _ptr;

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

class I8COnesorView
    with Onesor<int>, I8Onesor, ListMixin<int>, COnesor<int>, I8COnesor
    implements I8COnesor, COnesorView<int> {
  final I8COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  I8COnesorView(this._list, this.offset, this.length);

  @override
  late final ffi.Pointer<ffi.Int8> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }
}
