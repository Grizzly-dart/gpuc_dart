import 'dart:collection';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract class F64COnesor implements COnesor<double> {
  @override
  ffi.Pointer<ffi.Double> get ptr;

  @override
  List<double> asTypedList(int length);

  factory F64COnesor(ffi.Pointer<ffi.Double> ptr, int length,
          {Context? context}) =>
      _F64COnesor(ptr, length, context: context);

  static F64COnesor copy(Onesor<double> other, {Context? context}) =>
      _F64COnesor.copy(other, context: context);

  static F64COnesor fromList(List<double> list, {Context? context}) =>
      _F64COnesor.fromList(list, context: context);

  static F64COnesor sized(int length, {Context? context}) =>
      _F64COnesor.sized(length, context: context);

  static const sizeOfItem = 8;
}

class _F64COnesor
    with COnesorMixin<double>, F64COnesorMixin, ListMixin<double>
    implements F64COnesor {
  ffi.Pointer<ffi.Double> _ptr;

  int _length;

  _F64COnesor(this._ptr, this._length, {Context? context}) {
    assert(_ptr != ffi.nullptr);
    context?.add(this);
  }

  static _F64COnesor copy(Onesor<double> other, {Context? context}) {
    final clist = _F64COnesor.sized(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _F64COnesor fromList(List<double> list, {Context? context}) {
    final ret = _F64COnesor.sized(list.length, context: context);
    ret.ptr.asTypedList(list.length).setAll(0, list);
    return ret;
  }

  static _F64COnesor sized(int length, {Context? context}) {
    final ptr = ffi.calloc<ffi.Double>(length * F64COnesor.sizeOfItem);
    return _F64COnesor(ptr, length, context: context);
  }

  @override
  ffi.Pointer<ffi.Double> get ptr => _ptr;

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
    final newPtr = CFFI.realloc(_ptr.cast(), newLength * F64COnesor.sizeOfItem);
    if (newPtr == ffi.nullptr) {
      throw Exception('Failed to allocate memory');
    }
    _ptr = newPtr.cast();
    _length = newLength;
  }
}

class F64COnesorView
    with COnesorMixin<double>, F64COnesorMixin, ListMixin<double>
    implements F64COnesor, COnesorView<double> {
  final COnesor<double> _list;

  @override
  final int offset;

  @override
  final int length;

  F64COnesorView(this._list, this.offset, this.length);

  @override
  late final ffi.Pointer<ffi.Double> ptr =
      _list.ptr.cast<ffi.Double>() + offset;

  @override
  void release() {}

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot change length of view');
  }
}

mixin F64COnesorMixin implements COnesorMixin<double> {
  @override
  ffi.Pointer<ffi.Double> get ptr;

  @override
  List<double> asTypedList(int length) => ptr.asTypedList(length);

  @override
  int get lengthBytes => length * bytesPerItem;

  @override
  int get bytesPerItem => 8;

  @override
  double operator [](int index) => ptr[index];

  @override
  void operator []=(int index, double value) => ptr[index] = value;

  @override
  DeviceType get deviceType => DeviceType.c;

  @override
  int get deviceId => 0;

  @override
  double get defaultValue => 0;

  @override
  F64COnesor slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = F64COnesor.sized(length, context: context);
    CFFI.memcpy(ret.ptr.cast(), (ptr + start * F64COnesor.sizeOfItem).cast(),
        length * F64COnesor.sizeOfItem);
    return ret;
  }

  @override
  F64COnesor read({Context? context}) {
    final ret = F64COnesor.sized(length, context: context);
    CFFI.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  F64COnesorView view(int start, int length) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    if (this is COnesorView<double>) {
      start += (this as COnesorView<double>).offset;
    }
    return F64COnesorView(this, start, length);
  }
}
