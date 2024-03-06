import 'dart:io';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/clist.dart';

/*
  uint64_t totalGlobalMem;
  uint64_t totalConstMem;
  uint64_t sharedMemPerBlock;
  uint64_t reservedSharedMemPerBlock;
  uint64_t sharedMemPerMultiProcessor;
  uint32_t wrapSize;
  uint32_t multiProcessorCount;
  uint32_t maxThreadsPerMultiProcessor;
  uint32_t maxThreadsPerBlock;
  uint32_t maxBlocksPerMultiProcessor;
  uint32_t l2CacheSize;
  uint64_t memPitch;
  uint32_t memoryBusWidth;
  uint32_t pciBusID;
  uint32_t pciDeviceID;
  uint32_t pciDomainID;
 */

final class CudaDeviceProps extends ffi.Struct {
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

  Map<String, dynamic> toJson() => {
        'totalGlobalMem': totalGlobalMem,
        'totalConstMem': totalConstMem,
        'sharedMemPerBlock': sharedMemPerBlock,
        'reservedSharedMemPerBlock': reservedSharedMemPerBlock,
        'sharedMemPerMultiProcessor': sharedMemPerMultiProcessor,
        'warpSize': warpSize,
        'multiProcessorCount': multiProcessorCount,
        'maxThreadsPerMultiProcessor': maxThreadsPerMultiProcessor,
        'maxThreadsPerBlock': maxThreadsPerBlock,
        'maxBlocksPerMultiProcessor': maxBlocksPerMultiProcessor,
        'l2CacheSize': l2CacheSize,
        'memPitch': memPitch,
        'memoryBusWidth': memoryBusWidth,
        'pciBusID': pciBusID,
        'pciDeviceID': pciDeviceID,
        'pciDomainID': pciDomainID,
      };

  @override
  String toString() => toJson().toString();
}

typedef Op1D2Inp = void Function(ffi.Pointer<ffi.Void> out,
    ffi.Pointer<ffi.Void> inp1, ffi.Pointer<ffi.Void>, int size);
typedef Op1D2InpNative = ffi.Void Function(ffi.Pointer<ffi.Void> out,
    ffi.Pointer<ffi.Void> inp1, ffi.Pointer<ffi.Void>, ffi.Uint32 size);

typedef Op2DNative = ffi.Void Function(
    ffi.Pointer<ffi.Void> out, ffi.Pointer<ffi.Void> inp1, CSize2D);
typedef Op2D = void Function(
    ffi.Pointer<ffi.Void> out, ffi.Pointer<ffi.Void> inp1, CSize2D);

abstract class CudaFFIFunctions {
  static void initialize(ffi.DynamicLibrary dylib) {
    allocate = dylib.lookupFunction<
        ffi.Pointer<ffi.Void> Function(ffi.Uint64, ffi.Int32),
        ffi.Pointer<ffi.Void> Function(int, int)>('libtcCudaAlloc');
    release = dylib.lookupFunction<ffi.Void Function(ffi.Pointer<ffi.Void>),
        void Function(ffi.Pointer<ffi.Void>)>('libtcCudaFree');
    memcpy = dylib.lookupFunction<
        ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>,
            ffi.Uint64, ffi.Uint8, ffi.Int32),
        void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>, int, int,
            int)>('libtcCudaMemcpy');
    getDeviceProps = dylib.lookupFunction<CudaDeviceProps Function(ffi.Int32),
        CudaDeviceProps Function(int)>('libtcCudaGetDeviceProps');

    addition =
        dylib.lookupFunction<Op1D2InpNative, Op1D2Inp>('libtcCudaAddCkern');
    sum2D = dylib.lookupFunction<Op2DNative, Op2D>('libtcCudaSum2DCkern');
  }

  static late final ffi.Pointer<ffi.Void> Function(int size, int device)
      allocate;
  static late final void Function(ffi.Pointer<ffi.Void> ptr) release;

  static late final void Function(ffi.Pointer<ffi.Void> dst,
      ffi.Pointer<ffi.Void> src, int size, int dir, int deviceId) memcpy;

  static late final CudaDeviceProps Function(int device) getDeviceProps;

  static late final Op1D2Inp addition;
  static late final Op2D sum2D;

  /* TODO
  static void sum2D() {
    // TODO
  }
   */

  static late final void Function(
      ffi.Pointer<ffi.Double> out,
      ffi.Pointer<ffi.Double> inp,
      CSize2D kernS,
      CSize2D outS,
      CSize2D inS,
      CSize2D stride,
      CSize2D dialation,
      CSize2D padding,
      double padValue,
      int padMode) _maxpool2D;

  static void maxpool2D(Tensor out, Tensor inp, Size2D kernS,
      {Size2D stride = const Size2D(rows: 1, cols: 1),
      Size2D padding = const Size2D(rows: 0, cols: 0),
      double padValue = 0,
      PadMode padMode = PadMode.constant,
      Size2D dilation = const Size2D(rows: 1, cols: 1)}) {
    // TODO transfer to device if necessary instead?
    if (out.deviceType != DeviceType.cuda) {
      throw ArgumentError('Output tensor must be on CUDA device');
    }
    if (out.deviceType != inp.deviceType) {
      inp = inp.to(out.deviceType, deviceId: out.deviceId);
      // TODO release this after usage
    }

    final arena = ffi.Arena();
    try {
      final kernSPtr = CSize2D.fromSize2D(kernS, allocator: arena);
      final outSPtr = CSize2D.fromSize2D(out.size.twoD, allocator: arena);
      final inSPtr = CSize2D.fromSize2D(inp.size.twoD, allocator: arena);
      final strideSPtr = CSize2D.fromSize2D(stride, allocator: arena);
      final dilationSPtr = CSize2D.fromSize2D(dilation, allocator: arena);
      final paddingSPtr = CSize2D.fromSize2D(padding, allocator: arena);

      _maxpool2D(
          out.ptr,
          inp.ptr,
          kernSPtr.ref,
          outSPtr.ref,
          inSPtr.ref,
          strideSPtr.ref,
          dilationSPtr.ref,
          paddingSPtr.ref,
          padValue,
          padMode.index);
    } finally {
      arena.releaseAll();
    }
  }
}
