import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;

abstract class NList {
  int get length;
  int get lengthBytes;

  void release();

  // TODO implement partial write
  void copyFrom(NList src);

  // TODO implement partial read
  void copyTo(NList dst);

  CList read();
}

class DartList implements NList {
  final List<double> _list;

  DartList(this._list);

  @override
  int get length => _list.length;

  @override
  int get lengthBytes => length * 8;

  @override
  void release() {}

  @override
  void copyFrom(NList src) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is CList) {
      _list.setAll(0, src._mem.asTypedList(length));
      return;
    } else if (src is DartList) {
      _list.setAll(0, src._list);
      return;
    }
    final cSrc = src.read();
    try {
      _list.setAll(0, cSrc._mem.asTypedList(cSrc.length));
    } finally {
      cSrc.release();
    }
  }

  @override
  void copyTo(NList dst) {
    if (lengthBytes != dst.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (dst is CList) {
      dst._mem.asTypedList(dst.length).setAll(0, _list);
      return;
    } else if (dst is DartList) {
      dst._list.setAll(0, _list);
      return;
    }
    final cSrc = read();
    try {
      dst.copyFrom(cSrc);
    } finally {
      cSrc.release();
    }
  }

  @override
  CList read() {
    final clist = CList.allocate(_list.length);
    clist._mem.asTypedList(_list.length).setAll(0, _list);
    return clist;
  }
}

class CList implements NList {
  ffi.Pointer<ffi.Double> _mem;

  int _length;

  CList._(this._mem, this._length);

  static CList allocate(int length) {
    final mem = ffi.calloc<ffi.Double>(length * 8);
    return CList._(mem, length);
  }

  @override
  int get lengthBytes => length * 8;

  void resize(int length) {
    final newPtr = CListFFIFunctions.realloc(_mem, length * 8);
    if (newPtr == ffi.nullptr) {
      throw Exception('Failed to allocate memory');
    }
    _mem = newPtr.cast();
    _length = length;
  }

  @override
  int get length => _length;

  @override
  void release() {
    ffi.malloc.free(_mem);
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
      _mem.asTypedList(length).setAll(0, src._list);
      return;
    }
    src.copyTo(this);
  }

  @override
  void copyTo(NList list) {
    if (lengthBytes != list.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (list is CList) {
      CListFFIFunctions.memcpy(list._mem.cast(), _mem.cast(), lengthBytes);
      return;
    } else if (list is DartList) {
      list._list.setAll(0, _mem.asTypedList(length));
      return;
    }
    // TODO
  }

  @override
  CList read() {
    final clist = CList.allocate(length);
    CListFFIFunctions.memcpy(clist._mem.cast(), _mem.cast(), lengthBytes);
    return clist;
  }
}

abstract class CListFFIFunctions {
  static late final ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<void> oldPtr, int size) realloc;
  static late final void Function(
      ffi.Pointer<ffi.Void> dst, ffi.Pointer<ffi.Void> src, int size) memcpy;
}

class CudaList implements NList {
  final ffi.Pointer<ffi.Double> _mem;
  @override
  final int length;

  CudaList._(this._mem, this.length);

  @override
  int get lengthBytes => length * 8;

  static CudaList allocate(int length) {
    final mem = CudaFFIFunctions.allocate(length);
    return CudaList._(mem.cast(), length);
  }

  @override
  void release() => CudaFFIFunctions.release(_mem.cast());

  @override
  void copyFrom(NList src) {
    if(lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is CList) {
      CudaFFIFunctions.memcpy(_mem.cast(), src._mem.cast(), lengthBytes);
      return;
    }
    final cSrc = src.read();
    try {
      CudaFFIFunctions.memcpy(_mem.cast(), cSrc._mem.cast(), cSrc.length);
    } finally {
      cSrc.release();
    }
  }

  @override
  void copyTo(NList dst) {
    if (dst is CList) {
      CudaFFIFunctions.memcpy(dst._mem.cast(), dst._mem.cast(), dst.length);
      return;
    }
    final cSrc = read();
    try {
      dst.copyFrom(dst);
    } finally {
      cSrc.release();
    }
  }

  @override
  CList read() {
    final clist = CList.allocate(length);
    CudaFFIFunctions.memcpy(clist._mem.cast(), _mem.cast(), clist.lengthBytes);
    return clist;
  }
}

abstract class CudaFFIFunctions {
  static late final ffi.Pointer<ffi.Void> Function(int size) allocate;
  static late final void Function(ffi.Pointer<ffi.Void> ptr) release;

  static late final void Function(
      ffi.Pointer<ffi.Void> dst, ffi.Pointer<ffi.Void> data, int size) memcpy;
}
