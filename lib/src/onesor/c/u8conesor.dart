import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U8COnesor implements COnesor<int>, U8Onesor {
  @override
  ffi.Pointer<ffi.Uint8> get ptr;

  factory U8COnesor(ffi.Pointer<ffi.Uint8> ptr, int length,
          {Context? context}) =>
      _U8COnesor(ptr, length, context: context);

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
    CFFI.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  U8COnesor read({Context? context}) {
    final ret = U8COnesor.sized(length, context: context);
    CFFI.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  U8COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    if (this is COnesorView<int>) {
      start += (this as COnesorView<int>).offset;
    }
    return U8COnesorView(this, start, length);
  }
}

class _U8COnesor
    with Onesor<int>, U8Onesor, ListMixin<int>, COnesor<int>, U8COnesor
    implements U8COnesor {
  ffi.Pointer<ffi.Uint8> _ptr;

  int _length;

  _U8COnesor(this._ptr, this._length, {Context? context}) {
    assert(_ptr != ffi.nullptr);
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

  static _U8COnesor sized(int length, {Context? context}) {
    final ptr = ffi.calloc<ffi.Uint8>(length * Uint8List.bytesPerElement);
    return _U8COnesor(ptr, length, context: context);
  }

  @override
  ffi.Pointer<ffi.Uint8> get ptr => _ptr;

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

class U8COnesorView
    with Onesor<int>, U8Onesor, ListMixin<int>, COnesor<int>, U8COnesor
    implements U8COnesor, COnesorView<int> {
  final U8COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  U8COnesorView(this._list, this.offset, this.length);

  @override
  late final ffi.Pointer<ffi.Uint8> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }
}
