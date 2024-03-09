import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

mixin CListMixin implements CList {
  @override
  ffi.Pointer<ffi.Double> get ptr;

  @override
  void copyFrom(NList src) {
    if (lengthBytes != src.lengthBytes) {
      throw ArgumentError('Length mismatch');
    }
    if (src is CList) {
      CListFFI.memcpy(ptr.cast(), src.ptr.cast(), lengthBytes);
    } else if (src is DartList) {
      for (var i = 0; i < length; i++) {
        ptr.asTypedList(length).setAll(0, src.list);
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
      CListFFI.memcpy(dst.ptr.cast(), ptr.cast(), lengthBytes);
      return;
    } else if (dst is DartList) {
      dst.list.setAll(0, ptr.asTypedList(length));
      return;
    }
    dst.copyFrom(this);
  }

  @override
  CList read({Context? context}) {
    final ret = CList.allocate(length, context: context);
    CListFFI.memcpy(ret.ptr.cast(), ptr.cast(), lengthBytes);
    return ret;
  }

  @override
  CList slice(int start, int length, {Context? context}) {
    if (start > this.length) {
      throw ArgumentError('Start index out of range');
    } else if (start + length > this.length) {
      throw ArgumentError('Length out of range');
    }
    final ret = CList.allocate(length, context: context);
    CListFFI.memcpy(ret.ptr.cast(), (ptr + start * NList.byteSize).cast(),
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
    return CListView(this, start, length);
  }
}
