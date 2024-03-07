import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/tensor.dart';

final class CSize2D extends ffi.Struct {
  @ffi.Int32()
  external int r;

  @ffi.Int32()
  external int c;

  static ffi.Pointer<CSize2D> fromSize2D(Size2D size,
      {ffi.Allocator allocator = ffi.malloc}) {
    final cSize = allocator.allocate<CSize2D>(ffi.sizeOf<CSize2D>());
    cSize.ref.r = size.rows;
    cSize.ref.c = size.cols;
    return cSize;
  }
}

class CF64Ptr implements Resource {
  ffi.Pointer<ffi.Double> _mem;

  @override
  final Set<Context> contexts = {};

  CF64Ptr(this._mem, {Context? context}) {
    context?.add(this);
  }

  static CF64Ptr allocate({Context? context}) {
    final mem = ffi.calloc<ffi.Double>(8);
    return CF64Ptr(mem, context: context);
  }

  ffi.Pointer<ffi.Double> get ptr => _mem;

  double get value => _mem.value;

  set value(double value) => _mem.value = value;

  @override
  void release() {
    if (_mem == ffi.nullptr) {
      return;
    }
    ffi.malloc.free(_mem);
    _mem = ffi.nullptr;
  }

  @override
  void addContext(Context context) => contexts.add(context);

  @override
  void removeContext(Context context) => contexts.remove(context);
}