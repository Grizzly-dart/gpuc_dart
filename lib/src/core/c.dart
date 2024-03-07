import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/src/core/cuda.dart';
import 'package:gpuc_dart/src/core/dart_list.dart';
import 'package:gpuc_dart/src/core/releaseable.dart';

class CList extends NList {
  ffi.Pointer<ffi.Double> _mem;

  int _length;

  @override
  final Set<Context> contexts = {};

  CList._(this._mem, this._length, {Context? context}) {
    assert(_mem != ffi.nullptr);
    context?.add(this);
  }

  static CList copy(NList other, {Context? context}) {
    final clist = CList.allocate(other.length, context: context);
    clist.copyFrom(other);
    return clist;
  }

  static CList fromList(List<double> list, {Context? context}) {
    final clist = CList.allocate(list.length, context: context);
    clist._mem.asTypedList(list.length).setAll(0, list);
    return clist;
  }

  static CList allocate(int length, {Context? context}) {
    final mem = ffi.calloc<ffi.Double>(length * 8);
    return CList._(mem, length, context: context);
  }

  @override
  void addContext(Context context) => contexts.add(context);

  @override
  void removeContext(Context context) => contexts.remove(context);

  @override
  DeviceType get deviceType => DeviceType.c;

  @override
  int get deviceId => 0;

  @override
  int get length => _length;

  @override
  int get lengthBytes => length * 8;

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

  void resize(int length) {
    if (_mem == ffi.nullptr) {
      throw Exception('Memory already freed');
    }
    final newPtr = CListFFIFunctions.realloc(_mem.cast(), length * 8);
    if (newPtr == ffi.nullptr) {
      throw Exception('Failed to allocate memory');
    }
    _mem = newPtr.cast();
    _length = length;
  }

  @override
  void release() {
    if (_mem == ffi.nullptr) {
      return;
    }
    ffi.malloc.free(_mem);
    _mem = ffi.nullptr;
  }

  @override
  void copyFrom(NList src) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is CList) {
      CListFFIFunctions.memcpy(_mem.cast(), src._mem.cast(), lengthBytes);
      return;
    } else if (src is DartList) {
      _mem.asTypedList(length).setAll(0, src.list);
      return;
    }
    src.copyTo(this);
  }

  @override
  void copyTo(NList dst) {
    if (lengthBytes != dst.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (dst is CList) {
      CListFFIFunctions.memcpy(dst._mem.cast(), _mem.cast(), lengthBytes);
      return;
    } else if (dst is DartList) {
      dst.list.setAll(0, _mem.asTypedList(length));
      return;
    }
    dst.copyFrom(this);
  }

  @override
  CList read({Context? context}) {
    final clist = CList.allocate(length, context: context);
    CListFFIFunctions.memcpy(clist._mem.cast(), _mem.cast(), lengthBytes);
    return clist;
  }
}

abstract class CListFFIFunctions {
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

  static final finalizer = ffi.NativeFinalizer(CListFFIFunctions.freeNative);
}
