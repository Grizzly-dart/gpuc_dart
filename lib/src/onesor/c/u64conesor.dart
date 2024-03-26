import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class U64COnesor implements COnesor<int>, U64Onesor {
  @override
  ffi.Pointer<ffi.Uint64> get ptr;

  factory U64COnesor(ffi.Pointer<ffi.Uint64> ptr, int length,
          {Context? context}) =>
      _U64COnesor(ptr, length, context: context);

  static U64COnesor copy(Onesor<int> other, {Context? context}) =>
      _U64COnesor.copy(other, context: context);

  static U64COnesor fromList(List<int> list, {Context? context}) =>
      _U64COnesor.fromList(list, context: context);

  static U64COnesor sized(int length, {Context? context}) =>
      _U64COnesor.sized(length, context: context);

  @override
  List<int> asTypedList(int length) => ptr.asTypedList(length);

  @override
  int operator [](int index) => ptr[index];

  @override
  void operator []=(int index, int value) => ptr[index] = value;

  @override
  U64COnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = U64COnesor.sized(length, context: context);
    cffi!.memcpy(ret.ptr.cast(), (ptr + start * ret.bytesPerItem).cast(),
        length * ret.bytesPerItem);
    return ret;
  }

  @override
  U64COnesor read({Context? context}) {
    final ret = U64COnesor.sized(length, context: context);
    cffi!.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  U64COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U64COnesorView(this, start, length);
  }
}

class _U64COnesor
    with Onesor<int>, U64Onesor, ListMixin<int>, COnesor<int>, U64COnesor
    implements U64COnesor {
  ffi.Pointer<ffi.Uint64> _ptr;

  int _length;

  _U64COnesor(this._ptr, this._length, {Context? context}) {
    assert(_ptr != ffi.nullptr);
    context?.add(this);
  }

  static _U64COnesor copy(Onesor<int> other, {Context? context}) {
    final clist = _U64COnesor.sized(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _U64COnesor fromList(List<int> list, {Context? context}) {
    final ret = _U64COnesor.sized(list.length, context: context);
    ret.ptr.asTypedList(list.length).setAll(0, list);
    return ret;
  }

  static _U64COnesor sized(int length, {Context? context}) {
    final ptr = ffi.calloc<ffi.Uint64>(length * Uint64List.bytesPerElement);
    return _U64COnesor(ptr, length, context: context);
  }

  @override
  ffi.Pointer<ffi.Uint64> get ptr => _ptr;

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

class U64COnesorView
    with Onesor<int>, U64Onesor, ListMixin<int>, COnesor<int>, U64COnesor
    implements U64COnesor, COnesorView<int>, U64OnesorView {
  final U64COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  U64COnesorView(this._list, this.offset, this.length);

  @override
  late final ffi.Pointer<ffi.Uint64> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }

  @override
  U64COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    return U64COnesorView(_list, start + offset, length);
  }
}
