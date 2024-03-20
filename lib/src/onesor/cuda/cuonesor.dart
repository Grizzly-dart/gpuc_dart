import 'dart:ffi' as ffi;

import 'package:gpuc_dart/gpuc_dart.dart';

export 'f64cuonesor.dart';

abstract class CuOnesor<T extends num> extends Onesor<T> {
  ffi.Pointer<ffi.SizedNativeType> get ptr;
}