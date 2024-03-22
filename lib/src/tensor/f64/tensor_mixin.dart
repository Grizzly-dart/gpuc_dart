import 'package:gpuc_dart/gpuc_dart.dart';
import 'package:text_table/text_table.dart';

export 'tensor_mixin.dart';

mixin F64TensorMixin implements F64Tensor {
  // TODO start and length
  /*F64Tensor slice(/* Dim | int | Iterable<int> */ index, {Context? context}) {
    if (index is! Dim) index = Dim.from(index);
    if (size.isIndex(index)) {
      throw ArgumentError('Index out of range');
    }

    final outSize = Dim(size.asList.skip(index.dims));
    return F64Tensor(as1d.slice(index.nel * outSize.nel, outSize.nel), outSize,
        context: context);
  }*/

  @override
  F64Tensor operator [](dynamic /* Dim | int | Iterable<int> */ index) {
    if (index is! Dim) index = Dim.from(index);
    if (!size.isIndex(index)) {
      throw ArgumentError('Index out of range');
    }

    return OffsetF64TensorView(this, index);
  }

  @override
  void operator []=(
      dynamic /* Dim | int | Iterable<int> */ index, F64Tensor value) {
    if (index is! Dim) index = Dim.from(index);
    if (!size.isIndex(index)) {
      throw ArgumentError('Index out of range');
    }

    final outSize = Dim(size.asList.skip(index.dims));
    if (value.size.nel != outSize.nel) {
      throw ArgumentError('Size mismatch');
    }

    as1d.view(index.nel * outSize.nel, outSize.nel).copyFrom(value.as1d);
  }

  @override
  Onesor<double> row(int index, {int colDims = 1}) {
    if (size.dims < 2) {
      throw StateError('Must be at least a 2D tensor');
    }
    final size2d = size.squeeze2D(colDims: colDims);
    if (index < 0 || index >= size2d.rows) {
      throw ArgumentError('Index out of range');
    }
    return as1d.view(index * size2d.cols, size2d.cols);
  }

  /*
  @override
  void rearrange(List<int> order, {DeviceType? forceDeviceType}) {
    if (order.length != size.dims) {
      throw ArgumentError('Invalid order length');
    }
    final outSize = size.rearrange(order);
    // TODO detect device
    final deviceType = DeviceType.dart;
    if (deviceType == DeviceType.dart) {
      final outData = DartList.sized(outSize.nel);
      for (int i = 0; i < size.nel; i++) {
        final index = size.unravel(i);
        final outIndex = outSize.ravel(index.rearrange(order));
        outData[outIndex] = as1d[i];
      }
      final outTensor = Tensor(outData, outSize, name: 'rearrange($name)');
      return outTensor;
    } else if (deviceType == DeviceType.c) {
      // TODO
    } else if (deviceType == DeviceType.cuda) {
      /* TODO
      final outData = CList.sized(outSize.nel);
      final ctx = Context();
      try {
        int deviceId = 0; // TODO implement device selection
        final stream = CudaStream(deviceId, context: ctx);
        final inp = CudaList.copy(data, stream: stream, context: ctx);
        CudaFFI.rearrange(stream, outData.ptr.cast(), inp.ptr.cast(),
            _size.toList(), outSize.toList());
        final outTensor = Tensor(outData, outSize, name: 'rearrange($name)');
        ctx.releaseOnErr(outTensor);
        return outTensor;
      } catch (e) {
        ctx.release(isError: true);
        rethrow;
      } finally {
        ctx.release();
      }
       */
    }
    throw UnimplementedError('Device not implemented');
  }
   */

  @override
  void printTextTable({int precision = 4}) {
    for (int i = 0; i < size.numMatrices; i++) {
      print(TableRenderer().render(matrix(i)));
    }
  }
}
