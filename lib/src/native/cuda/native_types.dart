import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

class CudaFFI {
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

  final Op1d1Inp sin;
  final Op1d1Inp cos;
  final Op1d1Inp tan;
  final Op1d1Inp sinh;
  final Op1d1Inp cosh;
  final Op1d1Inp tanh;

  final Map<String, Op1d2Inp> additions;
  final Map<String, Op1d2Inp> subs;
  final Map<String, Op1d2Inp> muls;
  final Map<String, Op1d2Inp> divs;
  final Map<String, OpE> casts;

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

  final Op1d1InpS sigmoidActivation;
  final Op1d1InpS siluActivation;
  final StrPtr Function(ffi.Pointer<CCudaStream>, Ptr out, Ptr inp, int size,
      int beta, int threshold, int dataType) softplusActivation;
  final Op1d1InpS softsignActivation;
  final Op1d1InpS mishActivation;

  final StrPtr Function(ffi.Pointer<CCudaStream>, Ptr out, Ptr inp,
      Ptr threshold, Ptr value, int, int) minThreshold;

  CudaFFI({
    required this.getDeviceProps,
    required this.getMemInfo,
    required this.allocate,
    required this.memFree,
    required this.memcpy,
    required this.createStream,
    required this.destroyStream,
    required this.syncStream,
    required this.sin,
    required this.cos,
    required this.tan,
    required this.sinh,
    required this.cosh,
    required this.tanh,
    required this.additions,
    required this.subs,
    required this.muls,
    required this.divs,
    required this.casts,
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

  static CudaFFI? instance;

  static void initialize(ffi.DynamicLibrary dylib) {
    instance = CudaFFI.lookup(dylib);
  }

  factory CudaFFI.lookup(ffi.DynamicLibrary dylib) {
    final getDeviceProps = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaDeviceProps>, ffi.Int32),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaDeviceProps>, int)>('libtcCudaGetDeviceProps');
    final getMemInfo = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaMemInfo>, ffi.Int32),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaMemInfo>, int)>('libtcCudaGetMemInfo');

    final allocate = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Pointer<ffi.Void>>, ffi.Uint64),
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Pointer<ffi.Void>>, int)>('libtcCudaAlloc');
    final release = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>, ffi.Pointer<ffi.Void>),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>, ffi.Pointer<ffi.Void>)>('libtcCudaFree');
    final memcpy = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, ffi.Uint64),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>,
            ffi.Pointer<ffi.Void>,
            ffi.Pointer<ffi.Void>,
            int)>('libtcCudaMemcpy');

    final createStream = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>, ffi.Int32),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>, int)>('libtcCudaCreateStream');
    final destroyStream = dylib.lookupFunction<
        ffi.Pointer<ffi.Utf8> Function(ffi.Pointer<CCudaStream>),
        ffi.Pointer<ffi.Utf8> Function(
            ffi.Pointer<CCudaStream>)>('libtcCudaDestroyStream');
    final syncStream = dylib.lookupFunction<
            StrPtr Function(ffi.Pointer<CCudaStream>,
                ffi.Pointer<ffi.NativeFunction<ffi.Void Function(StrPtr)>>),
            StrPtr Function(ffi.Pointer<CCudaStream>,
                ffi.Pointer<ffi.NativeFunction<ffi.Void Function(StrPtr)>>)>(
        'libtcCudaSyncStream');

    final sin = dylib.lookupFunction<Op1d1InpNative, Op1d1Inp>('libtcCudaSin');
    final cos = dylib.lookupFunction<Op1d1InpNative, Op1d1Inp>('libtcCudaCos');
    final tan = dylib.lookupFunction<Op1d1InpNative, Op1d1Inp>('libtcCudaTan');
    final sinh =
        dylib.lookupFunction<Op1d1InpNative, Op1d1Inp>('libtcCudaSinh');
    final cosh =
        dylib.lookupFunction<Op1d1InpNative, Op1d1Inp>('libtcCudaCosh');
    final tanh =
        dylib.lookupFunction<Op1d1InpNative, Op1d1Inp>('libtcCudaTanh');

    final additions = <String, Op1d2Inp>{};
    final subs = <String, Op1d2Inp>{};
    final muls = <String, Op1d2Inp>{};
    final divs = <String, Op1d2Inp>{};
    final casts = <String, OpE>{};
    for (final o in NumType.values) {
      for (final inp1 in NumType.values) {
        for (final inp2 in NumType.values) {
          final key = '${o.short}_${inp1.short}_${inp2.short}';
          additions[key] = dylib
              .lookupFunction<Op1d2InpNative, Op1d2Inp>('libtcCudaAdd2_$key');
          subs[key] = dylib
              .lookupFunction<Op1d2InpNative, Op1d2Inp>('libtcCudaSub2_$key');
          muls[key] = dylib
              .lookupFunction<Op1d2InpNative, Op1d2Inp>('libtcCudaMul2_$key');
          divs[key] = dylib
              .lookupFunction<Op1d2InpNative, Op1d2Inp>('libtcCudaDiv2_$key');
        }
        if (o != inp1) {
          final key = '${o.short}_${inp1.short}';
          casts[key] =
              dylib.lookupFunction<OpENative, OpE>('libtcCudaCast_$key');
        }
      }
    }

    final sum2d = dylib.lookupFunction<Op2DNative, Op2d>('libtcCudaSum2d');
    final mean2d = dylib.lookupFunction<Op2DNative, Op2d>('libtcCudaMean2d');
    final variance2d = dylib
        .lookupFunction<Variance2DNative, Variance2d>('libtcCudaVariance2d');
    final normalize2d = dylib
        .lookupFunction<Normalize2DNative, Normalize2d>('libtcCudaNormalize2d');

    final transpose2d = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, CDim3),
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr,
            CDim3)>('libtcCudaTranspose2d');

    final pickRows = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, VoidPtr, VoidPtr, VoidPtr,
            CDim2, ffi.Uint8, ffi.Uint8),
        StrPtr Function(ffi.Pointer<CCudaStream>, VoidPtr, VoidPtr, VoidPtr,
            CDim2, int, int)>('libtcCudaPickRows');

    final matmul = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
            ffi.Uint32, ffi.Uint32, ffi.Uint32, ffi.Uint32),
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr, int,
            int, int, int)>('libtcCudaMatMul');
    final matmulT = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
            ffi.Uint32, ffi.Uint32, ffi.Uint32, ffi.Uint32),
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr, int,
            int, int, int)>('libtcCudaMatMulT');
    final matmulCadd = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
            F64Ptr, ffi.Uint32, ffi.Uint32, ffi.Uint32, ffi.Uint32),
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
            F64Ptr, int, int, int, int)>('libtcCudaMatMulCadd');
    final matmulTCadd = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
            F64Ptr, ffi.Uint32, ffi.Uint32, ffi.Uint32, ffi.Uint32),
        StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, F64Ptr,
            F64Ptr, int, int, int, int)>('libtcCudaMatMulTCadd');

    final maxPool2D =
        dylib.lookupFunction<MaxPool2DNative, MaxPool2D>('libtcCudaMaxPool2d');
    final conv2D =
        dylib.lookupFunction<Conv2DNative, Conv2D>('libtcCudaConv2d');

    final eluActivation = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, ffi.Uint64,
            ffi.Double, ffi.Uint8),
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, int, double,
            int)>('libtcCudaELU');
    final sigmoid =
        dylib.lookupFunction<Op1d1InpSNative, Op1d1InpS>('libtcCudaSigmoid');
    final silu =
        dylib.lookupFunction<Op1d1InpSNative, Op1d1InpS>('libtcCudaSiLU');
    final softplusActivation = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, ffi.Uint64,
            ffi.Uint8, ffi.Uint8, ffi.Uint8),
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, int, int, int,
            int)>('libtcCudaSoftplus');
    final softsignActivation = dylib.lookupFunction<Op1d1InpSNative, Op1d1InpS>(
        'libtcCudaSoftsign');
    final mishActivation = dylib
        .lookupFunction<Op1d1InpSNative, Op1d1InpS>('libtcCudaMish');
    final minThreshold = dylib.lookupFunction<
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, Ptr, Ptr,
            ffi.Uint64, ffi.Uint8),
        StrPtr Function(ffi.Pointer<CCudaStream>, Ptr, Ptr, Ptr, Ptr, int,
            int)>('libtcCudaMinThreshold');

    return CudaFFI(
      getDeviceProps: getDeviceProps,
      getMemInfo: getMemInfo,
      allocate: allocate,
      memFree: release,
      memcpy: memcpy,
      createStream: createStream,
      destroyStream: destroyStream,
      syncStream: syncStream,
      sin: sin,
      cos: cos,
      tan: tan,
      sinh: sinh,
      cosh: cosh,
      tanh: tanh,
      additions: additions,
      subs: subs,
      muls: muls,
      divs: divs,
      casts: casts,
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

typedef Op1d1InpS = StrPtr Function(
    ffi.Pointer<CCudaStream> stream, Ptr out, Ptr inp, int size, int dataType);
typedef Op1d1InpSNative = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp, ffi.Uint64 size, CNumType dataType);
typedef Op1d1Inp = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, int size, int outType, int inpType);
typedef Op1d1InpNative = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp, ffi.Uint64 size, CNumType outType, CNumType inpType);
typedef Op1d2Inp = StrPtr Function(
    ffi.Pointer<CCudaStream> stream, Ptr out, Ptr inp1, Ptr inp2, int size);
typedef Op1d2InpNative = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp1, Ptr inp2, ffi.Uint64 size);
typedef OpE = StrPtr Function(
    ffi.Pointer<CCudaStream> stream, Ptr out, Ptr inp, int size);
typedef OpENative = StrPtr Function(
    ffi.Pointer<CCudaStream> stream, Ptr out, Ptr inp, ffi.Uint64 size);

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
  external final ffi.Pointer<ffi.Void> stream;
  @ffi.Int32()
  external final int deviceId;

  Map<String, dynamic> toJson() => {
        'stream': stream,
        'deviceId': deviceId,
      };

  @override
  String toString() => toJson().toString();

  static ffi.Pointer<CCudaStream> create(int device) {
    final stream = ffi.calloc.allocate<CCudaStream>(ffi.sizeOf<CCudaStream>());
    try {
      final err = CudaFFI.instance!.createStream(stream, device);
      if (err != ffi.nullptr) {
        throw CudaException(err.toDartString());
      }
      // TODO setup finalizer
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
