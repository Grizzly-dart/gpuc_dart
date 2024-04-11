import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/native/c/c_types.dart';
import 'package:gpuc_dart/src/native/cuda/cu_types.dart';

class CuFFI {
  final ffi.DynamicLibrary lib;

  final StrPtr Function(ffi.Pointer<CCudaDeviceProps>, int device)
      getDeviceProps;
  final StrPtr Function(ffi.Pointer<CCudaMemInfo>, int device) getMemInfo;

  final StrPtr Function(ffi.Pointer<CCudaStream>,
      ffi.Pointer<ffi.Pointer<ffi.Void>>, int size) allocate;
  final StrPtr Function(ffi.Pointer<CCudaStream> ptr, ffi.Pointer<ffi.Void>)
      memFree;
  final StrPtr Function(ffi.Pointer<CCudaStream> stream,
      ffi.Pointer<ffi.Void> dst, ffi.Pointer<ffi.Void> src, int size) memcpy;

  final StrPtr Function(ffi.Pointer<CCudaStream> stream, int device)
      createStream;
  final StrPtr Function(ffi.Pointer<CCudaStream> stream) destroyStream;
  final StrPtr Function(ffi.Pointer<CCudaStream>,
      ffi.Pointer<ffi.NativeFunction<ffi.Void Function(StrPtr)>>) syncStream;

  late final CuOp1d1i2t cast =
      lib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuCast');
  late final CuOp1d1i1t neg =
      lib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuNeg');
  late final CuOp1d1i1t abs =
      lib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuAbs');
  late final CuOp1d1i2t sqr =
      lib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuSqr');
  late final CuOp1d1i2t sqrt =
      lib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuSqrt');
  late final CuOp1d1i2t log =
      lib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuLog');
  late final CuOp1d1i2t exp =
      lib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuExp');
  late final CuOp1d1i2t sin = lib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>(
      'tcuSin'); // TODO output dtype double?
  late final CuOp1d1i2t cos = lib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>(
      'tcuCos'); // TODO output dtype double?
  late final CuOp1d1i2t tan = lib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>(
      'tcuTan'); // TODO output dtype double?
  late final CuOp1d1i2t sinh = lib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>(
      'tcuSinh'); // TODO output dtype double?
  late final CuOp1d1i2t cosh = lib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>(
      'tcuCosh'); // TODO output dtype double?
  late final CuOp1d1i2t tanh = lib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>(
      'tcuTanh'); // TODO output dtype double?

  late final Map<String, CuOpBinary> plus = () {
    final ret = <String, CuOpBinary>{};
    // TODO
    throw UnimplementedError();
  }();
  late final CuOpBinary minus;
  late final CuOpBinary mul;
  late final CuOpBinary div;

  // TODO sum
  late final CuOp1d1i1t mean =
      lib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuMean');
  late final CuOp1d1i1t variance =
      lib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuVariance');

  late final CuOp2d sum2d = lib.lookupFunction<CunOp2D, CuOp2d>('tcuSum2d');
  late final CuOp2d mean2d = lib.lookupFunction<CunOp2D, CuOp2d>('tcuMean2d');
  late final CuVariance variance2d =
      lib.lookupFunction<CunVariance2D, CuVariance>('tcuVariance2d');
  late final CuNormalize2d normalize2d =
      lib.lookupFunction<CunNormalize2D, CuNormalize2d>('tcuNormalize2d');

  late final StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, CDim3)
      transpose2d = lib.lookupFunction<
          StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, CDim3),
          StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr,
              CDim3)>('tcuTranspose2d');

  late final StrPtr Function(
          ffi.Pointer<CCudaStream>, VoidPtr, VoidPtr, VoidPtr, CDim2, int, int)
      pickRows = lib.lookupFunction<
          StrPtr Function(ffi.Pointer<CCudaStream>, VoidPtr, VoidPtr, VoidPtr,
              CDim2, ffi.Uint8, ffi.Uint8),
          StrPtr Function(ffi.Pointer<CCudaStream>, VoidPtr, VoidPtr, VoidPtr,
              CDim2, int, int)>('tcuPickRows');

  late final StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
          int m, int n, int k, int batches) matmul =
      lib.lookupFunction<
          StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
              ffi.Uint32, ffi.Uint32, ffi.Uint32, ffi.Uint32),
          StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr, int,
              int, int, int)>('tcuMatMul');

  late final StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
          int m, int n, int k, int batches) matmulT =
      lib.lookupFunction<
          StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
              ffi.Uint32, ffi.Uint32, ffi.Uint32, ffi.Uint32),
          StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr, int,
              int, int, int)>('tcuMatMulT');

  late final StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
          F64Ptr, int m, int n, int k, int batches) matmulCadd =
      lib.lookupFunction<
          StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
              F64Ptr, ffi.Uint32, ffi.Uint32, ffi.Uint32, ffi.Uint32),
          StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
              F64Ptr, int, int, int, int)>('tcuMatMulCadd');

  late final StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
          F64Ptr, int m, int n, int k, int batches) matmulTCadd =
      lib.lookupFunction<
          StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
              F64Ptr, ffi.Uint32, ffi.Uint32, ffi.Uint32, ffi.Uint32),
          StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
              F64Ptr, int, int, int, int)>('tcuMatMulTCadd');

  late final CuMaxPool2D maxPool2D =
      lib.lookupFunction<CunMaxPool2D, CuMaxPool2D>('tcuMaxPool2d');
  late final CuConv2D conv2D =
      lib.lookupFunction<CunConv2D, CuConv2D>('tcuConv2d');

  late final StrPtr Function(
          ffi.Pointer<CCudaStream>, Ptr out, Ptr inp, int, double, int)
      eluActivation = lib.lookupFunction<
          StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, ffi.Uint64,
              ffi.Double, ffi.Uint8),
          StrPtr Function(
              ffi.Pointer<CCudaStream>, Ptr, Ptr, int, double, int)>('tcuELU');

  late final CuOp1d1i1t sigmoidActivation =
      lib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuSigmoid');
  late final CuOp1d1i1t siluActivation =
      lib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuSiLU');
  late final StrPtr Function(ffi.Pointer<CCudaStream>, Ptr out, Ptr inp,
          int size, int beta, int threshold, int dataType) softplusActivation =
      lib.lookupFunction<
          StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, ffi.Uint64,
              ffi.Uint8, ffi.Uint8, ffi.Uint8),
          StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, int, int, int,
              int)>('tcuSoftplus');
  late final CuOp1d1i1t softsignActivation =
      lib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuSoftsign');
  late final CuOp1d1i1t mishActivation =
      lib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuMish');

  late final StrPtr Function(ffi.Pointer<CCudaStream>, Ptr out, Ptr inp,
          Ptr threshold, Ptr value, int, int) minThreshold =
      lib.lookupFunction<
          StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, Ptr, Ptr,
              ffi.Uint64, ffi.Uint8),
          StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, Ptr, Ptr, int,
              int)>('tcuMinThreshold');

  CuFFI({
    required this.lib,
    required this.getDeviceProps,
    required this.getMemInfo,
    required this.allocate,
    required this.memFree,
    required this.memcpy,
    required this.createStream,
    required this.destroyStream,
    required this.syncStream,
  });

  void preload() {
    cast;
    neg;
    abs;
    sqr;
    sqrt;
    log;
    exp;
    sin;
    cos;
    tan;
    sinh;
    cosh;
    tanh;
    mean;
    variance;
    sum2d;
    mean2d;
    variance2d;
    normalize2d;
    transpose2d;
    pickRows;
    matmul;
    matmulT;
    matmulCadd;
    matmulTCadd;
    maxPool2D;
    conv2D;
    eluActivation;
    sigmoidActivation;
    siluActivation;
    softplusActivation;
    softsignActivation;
    mishActivation;
    minThreshold;
  }

  static CuFFI? instance;

  static void initialize(ffi.DynamicLibrary dylib) {
    instance = CuFFI.lookup(dylib);
  }

  factory CuFFI.lookup(ffi.DynamicLibrary lib) {
    final getDeviceProps = lib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaDeviceProps>, ffi.Int32),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaDeviceProps>, int)>('tcuGetDeviceProps');
    final getMemInfo = lib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaMemInfo>, ffi.Int32),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaMemInfo>, int)>('tcuGetMemInfo');

    final allocate = lib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Pointer<ffi.Void>>, ffi.Uint64),
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Pointer<ffi.Void>>, int)>('tcuAlloc');
    final release = lib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>, ffi.Pointer<ffi.Void>),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>, ffi.Pointer<ffi.Void>)>('tcuFree');
    final memcpy = lib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, ffi.Uint64),
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, int)>('tcuMemcpy');

    final createStream = lib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>, ffi.Int32),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>, int)>('tcuCreateStream');
    final destroyStream = lib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>)>('tcuDestroyStream');
    final syncStream = lib.lookupFunction<
            StrPtr Function(ffi.Pointer<CCudaStream>,
                ffi.Pointer<ffi.NativeFunction<ffi.Void Function(StrPtr)>>),
            StrPtr Function(ffi.Pointer<CCudaStream>,
                ffi.Pointer<ffi.NativeFunction<ffi.Void Function(StrPtr)>>)>(
        'tcuSyncStream');

    final plus = lib.lookupFunction<CunOpBinaryArith, CuOpBinary>('tcuPlus');
    final minus = lib.lookupFunction<CunOpBinaryArith, CuOpBinary>('tcuMinus');
    final mul = lib.lookupFunction<CunOpBinaryArith, CuOpBinary>('tcuMul');
    final div = lib.lookupFunction<CunOpBinaryArith, CuOpBinary>('tcuDiv');

    return CuFFI(
      lib: lib,
      getDeviceProps: getDeviceProps,
      getMemInfo: getMemInfo,
      allocate: allocate,
      memFree: release,
      memcpy: memcpy,
      createStream: createStream,
      destroyStream: destroyStream,
      syncStream: syncStream,
    );
  }
}
