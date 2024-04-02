import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

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

  final Op1d1i2t cast;
  final Op1d1i1t neg;
  final Op1d1i1t abs;
  final Op1d1i2t sqr;
  final Op1d1i2t sqrt;
  final Op1d1i2t log;
  final Op1d1i2t exp;
  final Op1d1i2t sin; // TODO output dtype double?
  final Op1d1i2t cos; // TODO output dtype double?
  final Op1d1i2t tan; // TODO output dtype double?
  final Op1d1i2t sinh; // TODO output dtype double?
  final Op1d1i2t cosh; // TODO output dtype double?
  final Op1d1i2t tanh; // TODO output dtype double?

  final OpBinaryArith plus;
  final OpBinaryArith minus;
  final OpBinaryArith mul;
  final OpBinaryArith div;

  // TODO sum
  final Op1d1i1t mean;
  final Op1d1i1t variance;

  final Op2d sum2d;
  final Op2d mean2d;
  final Variance2d variance2d;
  final Normalize2d normalize2d;

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

  final MaxPool2D maxPool2D;
  final Conv2D conv2D;

  final StrPtr Function(
          ffi.Pointer<CCudaStream>, Ptr out, Ptr inp, int, double, int)
      eluActivation;

  final Op1d1i1t sigmoidActivation;
  final Op1d1i1t siluActivation;
  final StrPtr Function(ffi.Pointer<CCudaStream>, Ptr out, Ptr inp, int size,
      int beta, int threshold, int dataType) softplusActivation;
  final Op1d1i1t softsignActivation;
  final Op1d1i1t mishActivation;

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

    final cast = dylib.lookupFunction<Op1d1i2tNative, Op1d1i2t>('tcuCast');
    final neg = dylib.lookupFunction<Op1d1i1tNative, Op1d1i1t>('tcuNeg');
    final abs = dylib.lookupFunction<Op1d1i1tNative, Op1d1i1t>('tcuAbs');
    final sqr = dylib.lookupFunction<Op1d1i2tNative, Op1d1i2t>('tcuSqr');
    final sqrt = dylib.lookupFunction<Op1d1i2tNative, Op1d1i2t>('tcuSqrt');
    final log = dylib.lookupFunction<Op1d1i2tNative, Op1d1i2t>('tcuLog');
    final exp = dylib.lookupFunction<Op1d1i2tNative, Op1d1i2t>('tcuExp');
    final sin = dylib.lookupFunction<Op1d1i2tNative, Op1d1i2t>('tcuSin');
    final cos = dylib.lookupFunction<Op1d1i2tNative, Op1d1i2t>('tcuCos');
    final tan = dylib.lookupFunction<Op1d1i2tNative, Op1d1i2t>('tcuTan');
    final sinh = dylib.lookupFunction<Op1d1i2tNative, Op1d1i2t>('tcuSinh');
    final cosh = dylib.lookupFunction<Op1d1i2tNative, Op1d1i2t>('tcuCosh');
    final tanh = dylib.lookupFunction<Op1d1i2tNative, Op1d1i2t>('tcuTanh');

    final plus =
        dylib.lookupFunction<OpBinaryArithNative, OpBinaryArith>('tcuPlus');
    final minus =
        dylib.lookupFunction<OpBinaryArithNative, OpBinaryArith>('tcuMinus');
    final mul =
        dylib.lookupFunction<OpBinaryArithNative, OpBinaryArith>('tcuMul');
    final div =
        dylib.lookupFunction<OpBinaryArithNative, OpBinaryArith>('tcuDiv');

    final mean = dylib.lookupFunction<Op1d1i1tNative, Op1d1i1t>('tcuMean');
    final variance =
        dylib.lookupFunction<Op1d1i1tNative, Op1d1i1t>('tcuVariance');

    final sum2d = dylib.lookupFunction<Op2DNative, Op2d>('tcuSum2d');
    final mean2d = dylib.lookupFunction<Op2DNative, Op2d>('tcuMean2d');
    final variance2d =
        dylib.lookupFunction<Variance2DNative, Variance2d>('tcuVariance2d');
    final normalize2d =
        dylib.lookupFunction<Normalize2DNative, Normalize2d>('tcuNormalize2d');

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
        dylib.lookupFunction<MaxPool2DNative, MaxPool2D>('tcuMaxPool2d');
    final conv2D = dylib.lookupFunction<Conv2DNative, Conv2D>('tcuConv2d');

    final eluActivation = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, ffi.Uint64,
            ffi.Double, ffi.Uint8),
        StrPtr Function(
            ffi.Pointer<CCudaStream>, Ptr, Ptr, int, double, int)>('tcuELU');
    final sigmoid =
        dylib.lookupFunction<Op1d1i1tNative, Op1d1i1t>('tcuSigmoid');
    final silu = dylib.lookupFunction<Op1d1i1tNative, Op1d1i1t>('tcuSiLU');
    final softplusActivation = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, ffi.Uint64,
            ffi.Uint8, ffi.Uint8, ffi.Uint8),
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, int, int, int,
            int)>('tcuSoftplus');
    final softsignActivation =
        dylib.lookupFunction<Op1d1i1tNative, Op1d1i1t>('tcuSoftsign');
    final mishActivation =
        dylib.lookupFunction<Op1d1i1tNative, Op1d1i1t>('tcuMish');
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

typedef CNumType = ffi.Uint8;

typedef Op1d1i1t = StrPtr Function(
    ffi.Pointer<CCudaStream> stream, Ptr out, Ptr inp, int size, int dataType);
typedef Op1d1i1tNative = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp, ffi.Uint64 size, CNumType dataType);

typedef Op1d1i2t = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, int size, int outType, int inpType);
typedef Op1d1i2tNative = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp, ffi.Uint64 size, CNumType outType, CNumType inpType);

typedef OpBinaryArith = StrPtr Function(
    ffi.Pointer<CCudaStream> stream,
    Ptr out,
    Ptr inp1,
    Ptr inp2,
    Ptr scalar,
    int size,
    int flipScalar,
    int outType,
    int inp1Type,
    int inp2Type);
typedef OpBinaryArithNative = StrPtr Function(
    ffi.Pointer<CCudaStream> stream,
    Ptr out,
    Ptr inp1,
    Ptr inp2,
    Ptr scalar,
    ffi.Uint64 size,
    ffi.Uint8 flipScalar,
    CNumType outType,
    CNumType inp1Type,
    CNumType inp2Type);

typedef Op1dF64Red = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, CDim2, int outType, int inpType);
typedef Op1dF64RedNative = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp, CDim2, CNumType, CNumType);

typedef Op2d = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, CDim2, int outType, int inpType);
typedef Op2DNative = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, CDim2, CNumType, CNumType);

typedef Variance2d = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, CDim2, int correction, int calcStd, int outType, int inpType);
typedef Variance2DNative = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp, CDim2, ffi.Uint64, ffi.Uint8, CNumType, CNumType);

typedef Normalize2d = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, CDim2, double epsilon, int outType, int inpType);
typedef Normalize2DNative = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp, CDim2, ffi.Double, ffi.Uint8, ffi.Uint8);

// TODO take dtype
typedef MaxPool2D = ffi.Pointer<ffi.Utf8> Function(
  ffi.Pointer<CCudaStream>,
  ffi.Pointer,
  ffi.Pointer,
  CDim2, // kernS
  CDim2, // outS
  CDim2, // inpS
  int, // matrices
  CDim2, // padding
  CDim2, // stride
  CDim2, // dilation
);
typedef MaxPool2DNative = ffi.Pointer<ffi.Utf8> Function(
  ffi.Pointer<CCudaStream>,
  ffi.Pointer,
  ffi.Pointer,
  CDim2, // kernS
  CDim2, // outS
  CDim2, // inpS
  ffi.Uint32, // matrices
  CDim2, // padding
  CDim2, // stride
  CDim2, // dilation
);

typedef Conv2DNative = ffi.Pointer<ffi.Utf8> Function(
  ffi.Pointer<CCudaStream>,
  ffi.Pointer<ffi.Double>, // out
  ffi.Pointer<ffi.Double>, // inp
  ffi.Pointer<ffi.Double>, // kernel
  ffi.Uint32, // batches
  CDim3, // outS
  CDim3, // inpS
  CDim2, // kernS
  ffi.Uint32, // groups
  CDim2, // padding
  ffi.Uint8, // padMode
  ffi.Double, // pad
  CDim2, // stride
  CDim2, // dilation
);
typedef Conv2D = ffi.Pointer<ffi.Utf8> Function(
  ffi.Pointer<CCudaStream>,
  ffi.Pointer<ffi.Double>, // out
  ffi.Pointer<ffi.Double>, // inp
  ffi.Pointer<ffi.Double>, // kernel
  int, // batches
  CDim3, // outS
  CDim3, // inpS
  CDim2, // kernS
  int, // groups
  CDim2, // padding
  int, // padMode
  double, // pad
  CDim2, // stride
  CDim2, // dilation
);

final class CCudaDeviceProps extends ffi.Struct {
  @ffi.Uint64()
  external int get totalGlobalMem;

  @ffi.Uint64()
  external int get totalConstMem;

  @ffi.Uint64()
  external int get sharedMemPerBlock;

  @ffi.Uint64()
  external int get reservedSharedMemPerBlock;

  @ffi.Uint64()
  external int get sharedMemPerMultiProcessor;

  @ffi.Uint32()
  external int get warpSize;

  @ffi.Uint32()
  external int get multiProcessorCount;

  @ffi.Uint32()
  external int get maxThreadsPerMultiProcessor;

  @ffi.Uint32()
  external int get maxThreadsPerBlock;

  @ffi.Uint32()
  external int get maxBlocksPerMultiProcessor;

  @ffi.Uint32()
  external int get l2CacheSize;

  @ffi.Uint64()
  external int get memPitch;

  @ffi.Uint32()
  external int get memoryBusWidth;

  @ffi.Uint32()
  external int get pciBusID;

  @ffi.Uint32()
  external int get pciDeviceID;

  @ffi.Uint32()
  external int get pciDomainID;
}

final class CCudaStream extends ffi.Struct {
  external ffi.Pointer<ffi.Void> stream;
  @ffi.Int32()
  external int deviceId;

  Map<String, dynamic> toJson() => {
        'stream': stream,
        'deviceId': deviceId,
      };

  @override
  String toString() => toJson().toString();

  static ffi.Pointer<CCudaStream> create(int device) {
    final stream = ffi.calloc.allocate<CCudaStream>(ffi.sizeOf<CCudaStream>());
    try {
      final err = CuFFI.instance!.createStream(stream, device);
      if (err != ffi.nullptr) {
        throw CudaException(err.toDartString());
      }
      return stream;
    } catch (e) {
      ffi.calloc.free(stream);
      rethrow;
    }
  }
}

final class CCudaMemInfo extends ffi.Struct {
  @ffi.Uint64()
  external int get free;

  @ffi.Uint64()
  external int get total;
}
