import 'dart:collection';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract mixin class I64COnesor implements COnesor<int>, I64Onesor {
  @override
  ffi.Pointer<ffi.Int64> get ptr;

  factory I64COnesor(ffi.Pointer<ffi.Int64> ptr, int length,
          {Context? context}) =>
      _I64COnesor(ptr, length, context: context);

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
    if (this is COnesorView<int>) {
      start += (this as COnesorView<int>).offset;
    }
    return I64COnesorView(this, start, length);
  }
}

class _I64COnesor
    with Onesor<int>, ListMixin<int>, COnesor<int>, I64Onesor, I64COnesor
    implements I64Onesor, I64COnesor {
  ffi.Pointer<ffi.Int64> _ptr;

  int _length;

  _I64COnesor(this._ptr, this._length, {Context? context}) {
    assert(_ptr != ffi.nullptr);
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

  static _I64COnesor sized(int length, {Context? context}) {
    final ptr = ffi.calloc<ffi.Int64>(length * Int64List.bytesPerElement);
    return _I64COnesor(ptr, length, context: context);
  }

  @override
  ffi.Pointer<ffi.Int64> get ptr => _ptr;

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

class I64COnesorView
    with Onesor<int>, I64Onesor, ListMixin<int>, COnesor<int>, I64COnesor
    implements I64COnesor, COnesorView<int> {
  final I64COnesor _list;

  @override
  final int offset;

  @override
  final int length;

  I64COnesorView(this._list, this.offset, this.length);

  @override
  late final ffi.Pointer<ffi.Int64> ptr = _list.ptr + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }
}
