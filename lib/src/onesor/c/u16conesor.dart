import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U16COnesor implements COnesor<int>, U16Onesor {
  @override
  ffi.Pointer<ffi.Uint16> get ptr;

  factory U16COnesor(ffi.Pointer<ffi.Uint16> ptr, int length,
          {Context? context}) =>
      _U16COnesor(ptr, length, context: context);

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
    cffi!.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  U16COnesor read({Context? context}) {
    final ret = U16COnesor.sized(length, context: context);
    cffi!.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  U16COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    if (this is COnesorView<int>) {
      start += (this as COnesorView<int>).offset;
    }
    return U16COnesorView(this, start, length);
  }
}

class _U16COnesor
    with Onesor<int>, U16Onesor, ListMixin<int>, COnesor<int>, U16COnesor
    implements U16COnesor {
  ffi.Pointer<ffi.Uint16> _ptr;

  int _length;

  _U16COnesor(this._ptr, this._length, {Context? context}) {
    assert(_ptr != ffi.nullptr);
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

  static _U16COnesor sized(int length, {Context? context}) {
    final ptr = ffi.calloc<ffi.Uint16>(length * Uint16List.bytesPerElement);
    return _U16COnesor(ptr, length, context: context);
  }

  @override
  ffi.Pointer<ffi.Uint16> get ptr => _ptr;

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

class U16COnesorView
    with Onesor<int>, U16Onesor, ListMixin<int>, COnesor<int>, U16COnesor
    implements U16COnesor, COnesorView<int> {
  final U16COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  U16COnesorView(this._list, this.offset, this.length);

  @override
  late final ffi.Pointer<ffi.Uint16> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }
}
