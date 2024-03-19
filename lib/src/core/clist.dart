import 'dart:collection';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract class CList implements NList {
  factory CList.copy(NList other, {Context? context}) =>
      _CListImpl.copy(other, context: context);

  factory CList.fromList(List<double> list, {Context? context}) =>
      _CListImpl.fromList(list, context: context);

  factory CList.sized(int length, {Context? context}) =>
      _CListImpl.sized(length, context: context);

  @override
  CList slice(int start, int length, {Context? context});

  @override
  CListView view(int start, int length);
}

class _CListImpl extends NList
    with CListMixin, ListMixin<double>
    implements CList {
  ffi.Pointer<ffi.Double> _mem;

  int _length;

  _CListImpl._(this._mem, this._length, {Context? context}) {
    assert(_mem != ffi.nullptr);
    context?.add(this);
  }

  static _CListImpl copy(NList other, {Context? context}) {
    final clist = _CListImpl.sized(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _CListImpl fromList(List<double> list, {Context? context}) {
    final clist = _CListImpl.sized(list.length, context: context);
    clist._mem.asTypedList(list.length).setAll(0, list);
    return clist;
  }

  static _CListImpl sized(int length, {Context? context}) {
    final mem = ffi.calloc<ffi.Double>(length * 8);
    return _CListImpl._(mem, length, context: context);
  }

  @override
  DeviceType get deviceType => DeviceType.c;

  @override
  int get deviceId => 0;

  @override
  int get length => _length;

  @override
  int get lengthBytes => length * NList.byteSize;

  @override
  double operator [](int index) {
    return _mem[index];
  }

  @override
  void operator []=(int index, double value) {
    _mem[index] = value;
  }

  @override
  ffi.Pointer<ffi.Double> get ptr => _mem;

  @override
  void release() {
    if (_mem == ffi.nullptr) {
      return;
    }
    ffi.malloc.free(_mem);
    _mem = ffi.nullptr;
  }

  @override
  set length(int newLength) {
    if (_mem == ffi.nullptr) {
      throw Exception('Memory already freed');
    }
    final newPtr = CFFI.realloc(_mem.cast(), newLength * NList.byteSize);
    if (newPtr == ffi.nullptr) {
      throw Exception('Failed to allocate memory');
    }
    _mem = newPtr.cast();
    _length = newLength;
  }
}

mixin CListMixin implements CList {
  @override
  ffi.Pointer<ffi.Double> get ptr;

  @override
  void copyFrom(NList src) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is CList) {
      CFFI.memcpy(ptr.cast(), src.ptr.cast(), lengthBytes);
    } else if (src is DartList) {
      for (var i = 0; i < length; i++) {
        ptr.asTypedList(length).setAll(0, src);
      }
    }
    src.copyTo(this);
  }

  @override
  void copyTo(NList dst) {
    if (lengthBytes != dst.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (dst is CList) {
      CFFI.memcpy(dst.ptr.cast(), ptr.cast(), lengthBytes);
      return;
    } else if (dst is DartList) {
      dst.setAll(0, ptr.asTypedList(length));
      return;
    }
    dst.copyFrom(this);
  }

  @override
  CList read({Context? context}) {
    final ret = CList.sized(length, context: context);
    CFFI.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  CList slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = CList.sized(length, context: context);
    CFFI.memcpy(ret.ptr.cast(), (ptr + start * NList.byteSize).cast(),
        length * NList.byteSize);
    return ret;
  }

  @override
  CListView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    if (this is CListView) {
      start += (this as CListView)._offset;
    }
    return CListView(this, start, length);
  }
}

class CListView extends NList
    with CListMixin, ListMixin<double>
    implements CList {
  final CList _list;

  final int _offset;

  final int _length;

  CListView(this._list, this._offset, this._length);

  @override
  DeviceType get deviceType => _list.deviceType;

  @override
  int get deviceId => _list.deviceId;

  @override
  int get length => _length;

  @override
  int get lengthBytes => _length * NList.byteSize;

  @override
  double operator [](int index) => _list[_offset + index];

  @override
  void operator []=(int index, double value) => _list[_offset + index] = value;

  @override
  ffi.Pointer<ffi.Double> get ptr => _list.ptr + _offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }
}
