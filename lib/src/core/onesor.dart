import 'dart:ffi' as ffi;
import 'package:gpuc_dart/gpuc_dart.dart';

abstract class Onesor<T> implements Resource, List<T> {
  Device get device;

  int get lengthBytes;

  ffi.Pointer<ffi.SizedNativeType> get ptr;

  // TODO subview

  // TODO implement partial write
  void copyFrom(Onesor<T> src);

  // TODO implement partial read
  void copyTo(Onesor<T> dst);

  CList read({Context? context});

  NList view(int start, int length);

  NList slice(int start, int length, {Context? context});

  @override
  List<T> toList({bool growable = true}) {
    final list = List<double>.filled(length, 0, growable: growable);
    copyTo(DartList.own(list));
    return list;
  }

  @override
  void release();

/*
  static NList allocate(int length,
      {DeviceType deviceType = DeviceType.c,
      int deviceId = 0,
      Context? context}) {
    switch (deviceType) {
      case DeviceType.c:
        return CList.allocate(length, context: context);
      case DeviceType.dart:
        return DartList.fromList(List.filled(length, 0));
      case DeviceType.cuda:
        return CudaList.allocate(length, deviceId: 0, context: context);
      case DeviceType.rocm:
        throw UnimplementedError('ROCm not implemented');
      case DeviceType.sycl:
        throw UnimplementedError('SYCL not implemented');
    }
  }
   */

/*
  static NList copy(NList other,
      {DeviceType device = DeviceType.c, int deviceId = 0, Context? context}) {
    switch (device) {
      case DeviceType.c:
        return CList.copy(other, context: context);
      case DeviceType.dart:
        return DartList.copy(other);
      case DeviceType.cuda:
        return CudaList.copy(other, deviceId: deviceId, context: context);
      case DeviceType.rocm:
        throw UnimplementedError('ROCm not implemented');
      case DeviceType.sycl:
        throw UnimplementedError('SYCL not implemented');
    }
  }
   */

  static const int byteSize = 8;
}