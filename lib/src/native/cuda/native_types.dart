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

  final Op1D2Inp addition;
  final Op1D2Inp subtract;
  final Op1D2Inp multiply;
  final Op1D2Inp divide;
  final Op2D sum2D;

  final StrPtr Function(ffi.Pointer<CCudaStream>, F64Ptr, F64Ptr, CDim3)
      transpose2D;

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

  CudaFFI({
    required this.getDeviceProps,
    required this.getMemInfo,
    required this.allocate,
    required this.memFree,
    required this.memcpy,
    required this.createStream,
    required this.destroyStream,
    required this.syncStream,
    required this.addition,
    required this.subtract,
    required this.multiply,
    required this.divide,
    required this.sum2D,
    required this.transpose2D,
    required this.pickRows,
    required this.matmul,
    required this.matmulT,
    required this.matmulCadd,
    required this.matmulTCadd,
    required this.maxPool2D,
    required this.conv2D,
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

    final addition =
        dylib.lookupFunction<Op1D2InpNative, Op1D2Inp>('libtcCudaAdd2');
    final subtract =
        dylib.lookupFunction<Op1D2InpNative, Op1D2Inp>('libtcCudaSubtract2');
    final multiply =
        dylib.lookupFunction<Op1D2InpNative, Op1D2Inp>('libtcCudaMultiply2');
    final divide =
        dylib.lookupFunction<Op1D2InpNative, Op1D2Inp>('libtcCudaDivide2');
    final sum2D = dylib.lookupFunction<Op2DNative, Op2D>('libtcCudaSum2D');

    final transpose2D = dylib.lookupFunction<
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
        dylib.lookupFunction<MaxPool2DNative, MaxPool2D>('libtcCudaMaxPool2D');
    final conv2D =
        dylib.lookupFunction<Conv2DNative, Conv2D>('libtcCudaConv2D');

    return CudaFFI(
      getDeviceProps: getDeviceProps,
      getMemInfo: getMemInfo,
      allocate: allocate,
      memFree: release,
      memcpy: memcpy,
      createStream: createStream,
      destroyStream: destroyStream,
      syncStream: syncStream,
      addition: addition,
      subtract: subtract,
      multiply: multiply,
      divide: divide,
      sum2D: sum2D,
      transpose2D: transpose2D,
      pickRows: pickRows,
      matmul: matmul,
      matmulT: matmulT,
      matmulCadd: matmulCadd,
      matmulTCadd: matmulTCadd,
      maxPool2D: maxPool2D,
      conv2D: conv2D,
    );
  }
}

typedef Op1D2Inp = StrPtr Function(ffi.Pointer<CCudaStream> stream, VoidPtr out,
    VoidPtr inp1, VoidPtr inp2, int size);
typedef Op1D2InpNative = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    VoidPtr out, VoidPtr inp1, VoidPtr inp2, ffi.Uint32 size);

typedef Op2D = StrPtr Function(
    ffi.Pointer<CCudaStream> stream, VoidPtr out, VoidPtr inp1, CDim2);
typedef Op2DNative = StrPtr Function(
    ffi.Pointer<CCudaStream> stream, VoidPtr out, VoidPtr inp1, CDim2);

typedef MaxPool2D = ffi.Pointer<ffi.Utf8> Function(
  ffi.Pointer<CCudaStream>,
  ffi.Pointer<ffi.Double>,
  ffi.Pointer<ffi.Double>,
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
  ffi.Pointer<ffi.Double>,
  ffi.Pointer<ffi.Double>,
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
