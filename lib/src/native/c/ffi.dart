import 'dart:ffi' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

class TensorCNativeFFI {
  final ffi
      .Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>
  freeNative;
  final void Function(ffi.Pointer<ffi.Void>) free;
  final ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void> oldPtr, int size)
  realloc;
  final void Function(
      ffi.Pointer<ffi.Void> dst, ffi.Pointer<ffi.Void> src, int size) memcpy;

  final List<List<List<COpBinaryArith>>> pluses;

  TensorCNativeFFI(
      {required this.freeNative,
        required this.free,
        required this.realloc,
        required this.memcpy,
        required this.pluses});

  factory TensorCNativeFFI.lookup(ffi.DynamicLibrary dylib) {
    final freeNative = dylib
        .lookup<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>(
        'tcFree');
    final free = dylib.lookupFunction<ffi.Void Function(ffi.Pointer<ffi.Void>),
        void Function(ffi.Pointer<ffi.Void>)>('tcFree');
    final realloc = dylib.lookupFunction<
        ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Uint64),
        ffi.Pointer<ffi.Void> Function(
            ffi.Pointer<ffi.Void>, int)>('tcRealloc');
    final memcpy = dylib.lookupFunction<
        ffi.Void Function(
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, ffi.Uint64),
        void Function(
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, int)>('tcMemcpy');

    final pluses = <List<List<COpBinaryArith>>>[];
    for (final oType in NumType.values) {
      final inp1List = <List<COpBinaryArith>>[];
      for (final inp1Type in NumType.values) {
        final inp2List = <COpBinaryArith>[];
        for (final inp2Type in NumType.values) {
          final func = dylib.lookupFunction<
              CnOpBinaryArith,
              COpBinaryArith>('_Z6tcPlusIdddEPKcPT_PKT0_PKT1_S9_yh');
          inp2List.add(func);
        }
        inp1List.add(inp2List);
      }
      pluses.add(inp1List);
    }

    return TensorCNativeFFI(
        freeNative: freeNative,
        free: free,
        realloc: realloc,
        memcpy: memcpy,
        pluses: pluses);
  }

  static void initialize(ffi.DynamicLibrary dylib) {
    instance = TensorCNativeFFI.lookup(dylib);
  }

  static TensorCNativeFFI? instance;
}

typedef COpBinaryArith = StrPtr Function(
    ffi.Pointer<ffi.Void> dst,
    ffi.Pointer<ffi.Void> src1,
    ffi.Pointer<ffi.Void> src2,
    ffi.Pointer<ffi.Void> scalar,
    int size,
    int flip,
    );
typedef CnOpBinaryArith = StrPtr Function(
    ffi.Pointer<ffi.Void> dst,
    ffi.Pointer<ffi.Void> src1,
    ffi.Pointer<ffi.Void> src2,
    ffi.Pointer<ffi.Void> scalar,
    ffi.Uint64 size,
    ffi.Uint8 flip,
    );