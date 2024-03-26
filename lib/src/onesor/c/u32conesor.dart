import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U32COnesor implements COnesor<int>, U32Onesor {
  @override
  ffi.Pointer<ffi.Uint32> get ptr;

  factory U32COnesor(ffi.Pointer<ffi.Uint32> ptr, int length,
          {Context? context}) =>
      _U32COnesor(ptr, length, context: context);

  static U32COnesor copy(Onesor<int> other, {Context? context}) =>
      _U32COnesor.copy(other, context: context);

  static U32COnesor fromList(List<int> list, {Context? context}) =>
      _U32COnesor.fromList(list, context: context);

  static U32COnesor sized(int length, {Context? context}) =>
      _U32COnesor.sized(length, context: context);

  @override
  List<int> asTypedList(int length) => ptr.asTypedList(length);

  @override
  int operator [](int index) => ptr[index];

  @override
  void operator []=(int index, int value) => ptr[index] = value;

  @override
  U32COnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = U32COnesor.sized(length, context: context);
    cffi!.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  U32COnesor read({Context? context}) {
    final ret = U32COnesor.sized(length, context: context);
    cffi!.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  U32COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U32COnesorView(this, start, length);
  }
}

class _U32COnesor
    with Onesor<int>, U32Onesor, ListMixin<int>, COnesor<int>, U32COnesor
    implements U32COnesor {
  ffi.Pointer<ffi.Uint32> _ptr;

  int _length;

  _U32COnesor(this._ptr, this._length, {Context? context}) {
    assert(_ptr != ffi.nullptr);
    context?.add(this);
  }

  static _U32COnesor copy(Onesor<int> other, {Context? context}) {
    final clist = _U32COnesor.sized(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _U32COnesor fromList(List<int> list, {Context? context}) {
    final ret = _U32COnesor.sized(list.length, context: context);
    ret.ptr.asTypedList(list.length).setAll(0, list);
    return ret;
  }

  static _U32COnesor sized(int length, {Context? context}) {
    final ptr = ffi.calloc<ffi.Uint32>(length * Uint32List.bytesPerElement);
    return _U32COnesor(ptr, length, context: context);
  }

  @override
  ffi.Pointer<ffi.Uint32> get ptr => _ptr;

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

class U32COnesorView
    with Onesor<int>, U32Onesor, ListMixin<int>, COnesor<int>, U32COnesor
    implements U32COnesor, COnesorView<int>, U32OnesorView {
  final U32COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  U32COnesorView(this._list, this.offset, this.length);

  @override
  late final ffi.Pointer<ffi.Uint32> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }

  @override
  U32COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U32COnesorView(_list, start + offset, length);
  }
}
