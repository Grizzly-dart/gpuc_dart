import 'dart:math';

import 'package:gpuc_dart/gpuc_dart.dart';

extension CudaSplitExtension on Cuda {
  // TODO use tensor views instead
  Future<Tensor> splitMatmul(int deviceId, Tensor a, Tensor b,
      {Tensor? out}) async {
    if (a.size.cols != b.size.rows) {
      throw ArgumentError('Columns of A must match rows of B');
    }
    int numMats = a.size.numMatrices;
    if (numMats != b.size.numMatrices) {
      throw ArgumentError('Number of matrices must match');
    }

    final inp1Size2D = a.size.to2D();
    final inp2Size2D = b.size.to2D();
    final outSize2D = Dim2(a.size.rows, b.size.cols);
    final outSize =
        outSize2D.extend2D(a.size.asList.take(a.size.asList.length - 2));

    final ctx = Context();
    try {
      if (out == null) {
        out = Tensor.sized(outSize, name: '${a.name} * ${b.name}');
        ctx.releaseOnErr(out);
      } else {
        if (out.size != outSize) {
          throw ArgumentError('Size mismatch');
        }
      }
      final props = cuda.getMemInfo(deviceId);
      int batchSize = props.total ~/
          ((inp1Size2D.nel + inp2Size2D.nel + outSize2D.nel) * 8);
      if (batchSize < 1) {
        throw StateError('Insufficient memory');
      } else if (batchSize > numMats) {
        batchSize = numMats;
      }
      final streams = <CudaStream>[];
      int batchStart = 0;
      while (batchStart < numMats) {
        int split = min(batchSize, numMats - batchStart);
        final stream = CudaStream(deviceId, context: ctx);
        streams.add(stream);
        final inp1 = CudaList.copy(
            a.as1d.view(batchStart * batchSize * inp1Size2D.nel,
                split * inp1Size2D.nel),
            stream: stream,
            context: ctx);
        final inp2 = CudaList.copy(
            b.as1d.view(batchStart * batchSize * inp2Size2D.nel,
                split * inp2Size2D.nel),
            stream: stream,
            context: ctx);
        final outMat =
            CudaList.sized(stream, split * outSize2D.nel, context: ctx);
        cuda.matmul(stream, outMat.ptr.cast(), inp1.ptr, inp2.ptr, a.size.rows,
            a.size.cols, b.size.cols, split);
        outMat.copyTo(
            out.as1d.view(
                batchStart * batchSize * outSize2D.nel, split * outSize2D.nel),
            stream: stream);
        inp1.release(stream: stream);
        inp2.release(stream: stream);
        outMat.release(stream: stream);
        batchStart += split;
      }
      await Future.wait(streams.map((s) => s.sync()));
      return out;
    } catch (e) {
      ctx.release(isError: true);
      rethrow;
    } finally {
      ctx.release();
    }
  }
}
