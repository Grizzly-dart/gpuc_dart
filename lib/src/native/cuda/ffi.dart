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

  final CuOp1d1i2t cast;
  final CuOp1d1i1t neg;
  final CuOp1d1i1t abs;
  final CuOp1d1i2t sqr;
  final CuOp1d1i2t sqrt;
  final CuOp1d1i2t log;
  final CuOp1d1i2t exp;
  final CuOp1d1i2t sin; // TODO output dtype double?
  final CuOp1d1i2t cos; // TODO output dtype double?
  final CuOp1d1i2t tan; // TODO output dtype double?
  final CuOp1d1i2t sinh; // TODO output dtype double?
  final CuOp1d1i2t cosh; // TODO output dtype double?
  final CuOp1d1i2t tanh; // TODO output dtype double?

  final CuOpBinary plus;
  final CuOpBinary minus;
  final CuOpBinary mul;
  final CuOpBinary div;

  // TODO sum
  final CuOp1d1i1t mean;
  final CuOp1d1i1t variance;

  final CuOp2d sum2d;
  final CuOp2d mean2d;
  final CuVariance variance2d;
  final CuNormalize2d normalize2d;

  final StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, CDim3)
      transpose2d;

  final StrPtr Function(
          ffi.Pointer<CCudaStream>, VoidPtr, VoidPtr, VoidPtr, CDim2, int, int)
      pickRows;

  final StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr, int m,
      int n, int k, int batches) matmul;

  final StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr, int m,
      int n, int k, int batches) matmulT;

  final StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
      F64Ptr, int m, int n, int k, int batches) matmulCadd;

  final StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
      F64Ptr, int m, int n, int k, int batches) matmulTCadd;

  final CuMaxPool2D maxPool2D;
  final CuConv2D conv2D;

  final StrPtr Function(
          ffi.Pointer<CCudaStream>, Ptr out, Ptr inp, int, double, int)
      eluActivation;

  final CuOp1d1i1t sigmoidActivation;
  final CuOp1d1i1t siluActivation;
  final StrPtr Function(ffi.Pointer<CCudaStream>, Ptr out, Ptr inp, int size,
      int beta, int threshold, int dataType) softplusActivation;
  final CuOp1d1i1t softsignActivation;
  final CuOp1d1i1t mishActivation;

  final StrPtr Function(ffi.Pointer<CCudaStream>, Ptr out, Ptr inp,
      Ptr threshold, Ptr value, int, int) minThreshold;

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
    required this.cast,
    required this.neg,
    required this.abs,
    required this.sqr,
    required this.sqrt,
    required this.log,
    required this.exp,
    required this.sin,
    required this.cos,
    required this.tan,
    required this.sinh,
    required this.cosh,
    required this.tanh,
    required this.plus,
    required this.minus,
    required this.mul,
    required this.div,
    required this.mean,
    required this.variance,
    required this.sum2d,
    required this.mean2d,
    required this.variance2d,
    required this.normalize2d,
    required this.transpose2d,
    required this.pickRows,
    required this.matmul,
    required this.matmulT,
    required this.matmulCadd,
    required this.matmulTCadd,
    required this.maxPool2D,
    required this.conv2D,
    required this.eluActivation,
    required this.sigmoidActivation,
    required this.siluActivation,
    required this.softsignActivation,
    required this.softplusActivation,
    required this.mishActivation,
    required this.minThreshold,
  });

  static CuFFI? instance;

  static void initialize(ffi.DynamicLibrary dylib) {
    instance = CuFFI.lookup(dylib);
  }

  factory CuFFI.lookup(ffi.DynamicLibrary dylib) {
    final getDeviceProps = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaDeviceProps>, ffi.Int32),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaDeviceProps>, int)>('tcuGetDeviceProps');
    final getMemInfo = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaMemInfo>, ffi.Int32),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaMemInfo>, int)>('tcuGetMemInfo');

    final allocate = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Pointer<ffi.Void>>, ffi.Uint64),
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Pointer<ffi.Void>>, int)>('tcuAlloc');
    final release = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>, ffi.Pointer<ffi.Void>),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>, ffi.Pointer<ffi.Void>)>('tcuFree');
    final memcpy = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, ffi.Uint64),
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, int)>('tcuMemcpy');

    final createStream = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>, ffi.Int32),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>, int)>('tcuCreateStream');
    final destroyStream = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>)>('tcuDestroyStream');
    final syncStream = dylib.lookupFunction<
            StrPtr Function(ffi.Pointer<CCudaStream>,
                ffi.Pointer<ffi.NativeFunction<ffi.Void Function(StrPtr)>>),
            StrPtr Function(ffi.Pointer<CCudaStream>,
                ffi.Pointer<ffi.NativeFunction<ffi.Void Function(StrPtr)>>)>(
        'tcuSyncStream');

    final cast = dylib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuCast');
    final neg = dylib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuNeg');
    final abs = dylib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuAbs');
    final sqr = dylib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuSqr');
    final sqrt = dylib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuSqrt');
    final log = dylib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuLog');
    final exp = dylib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuExp');
    final sin = dylib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuSin');
    final cos = dylib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuCos');
    final tan = dylib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuTan');
    final sinh = dylib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuSinh');
    final cosh = dylib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuCosh');
    final tanh = dylib.lookupFunction<CunOp1d1i2t, CuOp1d1i2t>('tcuTanh');

    final plus =
        dylib.lookupFunction<CunOpBinaryArith, CuOpBinary>('tcuPlus');
    final minus =
        dylib.lookupFunction<CunOpBinaryArith, CuOpBinary>('tcuMinus');
    final mul =
        dylib.lookupFunction<CunOpBinaryArith, CuOpBinary>('tcuMul');
    final div =
        dylib.lookupFunction<CunOpBinaryArith, CuOpBinary>('tcuDiv');

    final mean = dylib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuMean');
    final variance =
        dylib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuVariance');

    final sum2d = dylib.lookupFunction<CunOp2D, CuOp2d>('tcuSum2d');
    final mean2d = dylib.lookupFunction<CunOp2D, CuOp2d>('tcuMean2d');
    final variance2d =
        dylib.lookupFunction<CunVariance2D, CuVariance>('tcuVariance2d');
    final normalize2d =
        dylib.lookupFunction<CunNormalize2D, CuNormalize2d>('tcuNormalize2d');

    final transpose2d = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, CDim3),
        StrPtr Function(
            ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, CDim3)>('tcuTranspose2d');

    final pickRows = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, VoidPtr, VoidPtr, VoidPtr,
            CDim2, ffi.Uint8, ffi.Uint8),
        StrPtr Function(ffi.Pointer<CCudaStream>, VoidPtr, VoidPtr, VoidPtr,
            CDim2, int, int)>('tcuPickRows');

    final matmul = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
            ffi.Uint32, ffi.Uint32, ffi.Uint32, ffi.Uint32),
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr, int,
            int, int, int)>('tcuMatMul');
    final matmulT = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
            ffi.Uint32, ffi.Uint32, ffi.Uint32, ffi.Uint32),
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr, int,
            int, int, int)>('tcuMatMulT');
    final matmulCadd = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
            F64Ptr, ffi.Uint32, ffi.Uint32, ffi.Uint32, ffi.Uint32),
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
            F64Ptr, int, int, int, int)>('tcuMatMulCadd');
    final matmulTCadd = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
            F64Ptr, ffi.Uint32, ffi.Uint32, ffi.Uint32, ffi.Uint32),
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
            F64Ptr, int, int, int, int)>('tcuMatMulTCadd');

    final maxPool2D =
        dylib.lookupFunction<CunMaxPool2D, CuMaxPool2D>('tcuMaxPool2d');
    final conv2D = dylib.lookupFunction<CunConv2D, CuConv2D>('tcuConv2d');

    final eluActivation = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, ffi.Uint64,
            ffi.Double, ffi.Uint8),
        StrPtr Function(
            ffi.Pointer<CCudaStream>, Ptr, Ptr, int, double, int)>('tcuELU');
    final sigmoid =
        dylib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuSigmoid');
    final silu = dylib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuSiLU');
    final softplusActivation = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, ffi.Uint64,
            ffi.Uint8, ffi.Uint8, ffi.Uint8),
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, int, int, int,
            int)>('tcuSoftplus');
    final softsignActivation =
        dylib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuSoftsign');
    final mishActivation =
        dylib.lookupFunction<CunOp1d1i1t, CuOp1d1i1t>('tcuMish');
    final minThreshold = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, Ptr, Ptr,
            ffi.Uint64, ffi.Uint8),
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, Ptr, Ptr, int,
            int)>('tcuMinThreshold');

    return CuFFI(
      lib: dylib,
      getDeviceProps: getDeviceProps,
      getMemInfo: getMemInfo,
      allocate: allocate,
      memFree: release,
      memcpy: memcpy,
      createStream: createStream,
      destroyStream: destroyStream,
      syncStream: syncStream,
      cast: cast,
      neg: neg,
      abs: abs,
      sqr: sqr,
      sqrt: sqrt,
      log: log,
      exp: exp,
      sin: sin,
      cos: cos,
      tan: tan,
      sinh: sinh,
      cosh: cosh,
      tanh: tanh,
      plus: plus,
      minus: minus,
      mul: mul,
      div: div,
      mean: mean,
      variance: variance,
      sum2d: sum2d,
      mean2d: mean2d,
      variance2d: variance2d,
      normalize2d: normalize2d,
      transpose2d: transpose2d,
      pickRows: pickRows,
      matmul: matmul,
      matmulT: matmulT,
      matmulCadd: matmulCadd,
      matmulTCadd: matmulTCadd,
      maxPool2D: maxPool2D,
      conv2D: conv2D,
      eluActivation: eluActivation,
      sigmoidActivation: sigmoid,
      siluActivation: silu,
      softplusActivation: softplusActivation,
      softsignActivation: softsignActivation,
      mishActivation: mishActivation,
      minThreshold: minThreshold,
    );
  }
}
