import 'dart:collection';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

export 'clist_mixin.dart';
export 'clist_view.dart';

abstract class CList implements NList, List<double> {
  factory CList.copy(NList other, {Context? context}) =>
      _CListImpl.copy(other, context: context);

  factory CList.fromList(List<double> list, {Context? context}) =>
      _CListImpl.fromList(list, context: context);

  factory CList.allocate(int length, {Context? context}) =>
      _CListImpl.allocate(length, context: context);

  CListView view(int start, int length);

  CList slice(int start, int length, {Context? context});
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
    final clist = _CListImpl.allocate(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static _CListImpl fromList(List<double> list, {Context? context}) {
    final clist = _CListImpl.allocate(list.length, context: context);
    clist._mem.asTypedList(list.length).setAll(0, list);
    return clist;
  }

  static _CListImpl allocate(int length, {Context? context}) {
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
    final newPtr = CListFFI.realloc(_mem.cast(), newLength * NList.byteSize);
    if (newPtr == ffi.nullptr) {
      throw Exception('Failed to allocate memory');
    }
    _mem = newPtr.cast();
    _length = newLength;
  }
}

abstract class CListFFI {
  static late final ffi
      .Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>
      freeNative;
  static late final void Function(ffi.Pointer<ffi.Void>) free;
  static late final ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void> oldPtr, int size) realloc;
  static late final void Function(
      ffi.Pointer<ffi.Void> dst, ffi.Pointer<ffi.Void> src, int size) memcpy;

  static void initialize(ffi.DynamicLibrary dylib) {
    freeNative = dylib
        .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
            'libtcFree');
    free = dylib.lookupFunction<ffi.Void Function(ffi.Pointer<ffi.Void>),
        void Function(ffi.Pointer<ffi.Void>)>('libtcFree');
    realloc = dylib.lookupFunction<
        ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Uint64),
        ffi.Pointer<ffi.Void> Function(
            ffi.Pointer<ffi.Void>, int)>('libtcRealloc');
    memcpy = dylib.lookupFunction<
        ffi.Void Function(
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, ffi.Uint64),
        void Function(
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, int)>('libtcMemcpy');
  }

  static final finalizer = ffi.NativeFinalizer(CListFFI.freeNative);
}
