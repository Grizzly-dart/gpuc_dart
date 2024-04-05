import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:gpuc_dart/src/native/c/c_types.dart';

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

typedef CuOp1d1i1t = StrPtr Function(
    ffi.Pointer<CCudaStream> stream, Ptr out, Ptr inp, int size, int dataType);
typedef CunOp1d1i1t = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp, ffi.Uint64 size, CNumType dataType);

typedef CuOp1d1i2t = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, int size, int outType, int inpType);
typedef CunOp1d1i2t = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp, ffi.Uint64 size, CNumType outType, CNumType inpType);

typedef CuOpBinary = StrPtr Function(
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
typedef CunOpBinaryArith = StrPtr Function(
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

typedef CuOp1dF64Red = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, CDim2, int outType, int inpType);
typedef CunOp1dF64RedNative = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp, CDim2, CNumType, CNumType);

typedef CuOp2d = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, CDim2, int outType, int inpType);
typedef CunOp2D = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, CDim2, CNumType, CNumType);

typedef CuVariance = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, CDim2, int correction, int calcStd, int outType, int inpType);
typedef CunVariance2D = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp, CDim2, ffi.Uint64, ffi.Uint8, CNumType, CNumType);

typedef CuNormalize2d = StrPtr Function(ffi.Pointer<CCudaStream> stream, Ptr out,
    Ptr inp, CDim2, double epsilon, int outType, int inpType);
typedef CunNormalize2D = StrPtr Function(ffi.Pointer<CCudaStream> stream,
    Ptr out, Ptr inp, CDim2, ffi.Double, ffi.Uint8, ffi.Uint8);

// TODO take dtype
typedef CuMaxPool2D = ffi.Pointer<ffi.Utf8> Function(
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
typedef CunMaxPool2D = ffi.Pointer<ffi.Utf8> Function(
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

typedef CunConv2D = ffi.Pointer<ffi.Utf8> Function(
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
typedef CuConv2D = ffi.Pointer<ffi.Utf8> Function(
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