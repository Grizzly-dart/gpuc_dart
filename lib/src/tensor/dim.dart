import 'dart:collection';

import 'package:gpuc_dart/gpuc_dart.dart';

// TODO move this to grizzly?
abstract class Dim {
  factory Dim.from(value) {
    if (value is Dim) {
      return value;
    } else if (value is Iterable) {
      return Dim(value.cast());
    } else if (value is int) {
      return Dim([value]);
    } else if (value is ({int rows, int cols})) {
      return Dim.to2D(value.rows, value.cols);
    } else if (value is ({int r, int c})) {
      return Dim.to2D(value.r, value.c);
    }
    throw ArgumentError('Invalid type');
  }

  factory Dim(Iterable<int> sizes) => _DimImpl(List.from(sizes));

  factory Dim.to2D(int rows, [int? cols]) => _DimImpl([rows, cols ?? rows]);

  int operator [](int index);

  int get dims;

  Dim2 to2D();

  int get nel;

  int get rows;

  int get cols;

  int get channels;

  int get batch;

  int get numMatrices;

  bool isIndex(Dim other);

  Dim get strides;

  Dim unravel(int index);

  int get ravel;

  Dim reshape(Dim newSize);

  Dim reshapeDims(int dims);

  Dim2 squeeze2D({int colDims = 1});

  Dim squeezeFront(int dims);

  Dim squeeze(int dims);

  Dim rearrange(List<int> order);

  // TODO swap

  UnmodifiableListView<int> get asList;

  Dim incrementIndex(Dim index);

  Dim zeroIndex();

  List<int> toList();

  List<int> toJson() => toList();
}

mixin DimMixin implements Dim {
  @override
  Dim unravel(int index) {
    if (index < 0 || index >= nel) {
      throw ArgumentError('Index out of range');
    }
    final sizes = List<int>.filled(dims, 0);
    for (var i = dims - 1; i >= 0; i--) {
      sizes[i] = index % this[i];
      index ~/= this[i];
    }
    return Dim(sizes);
  }

  @override
  int get ravel {
    if (dims == 1) {
      return this[0];
    }
    int ret = 0;
    final strides = this.strides;
    for (int i = dims - 1; i >= 0; i--) {
      ret += this[i] * strides[i];
    }
    return ret;
  }

  @override
  bool isIndex(Dim other) {
    if (other.dims > dims) {
      return false;
    }
    for (var i = 0; i < other.dims; i++) {
      if (other[i] >= this[i]) {
        return false;
      }
    }
    return true;
  }

  @override
  Dim2 squeeze2D({int colDims = 1}) {
    if (colDims > dims) {
      throw ArgumentError('Dimension out of range');
    } else if (colDims == dims) {
      return Dim2(nel, 1);
    }
    int n = dims - colDims;
    return Dim2(asList.take(n).prod, asList.skip(n).prod);
  }

  @override
  Dim squeezeFront(int dims) {
    if (dims > this.dims) {
      throw ArgumentError('Dimension out of range');
    } else if (rows == this.dims) {
      return Dim([nel]);
    }
    return Dim([asList.take(dims).prod, ...asList.skip(dims)]);
  }

  @override
  Dim squeeze(int dims) {
    if (dims > this.dims) {
      throw ArgumentError('Dimension out of range');
    } else if (rows == this.dims) {
      return Dim([nel]);
    }
    int n = this.dims - dims;
    return Dim(asList.take(n).followedBy([asList.skip(n).prod]));
  }

  @override
  Dim rearrange(List<int> order) {
    if (order.toSet().length != dims) {
      throw ArgumentError('Invalid order length');
    }
    if (order.any((e) => e < 0 || e >= dims)) {
      throw ArgumentError('Invalid order index');
    }
    final sizes = List<int>.filled(dims, 0);
    for (var i = 0; i < dims; i++) {
      sizes[i] = this[order[i]];
    }
    return Dim(sizes);
  }

  @override
  Dim zeroIndex() => Dim(List.filled(dims, 0));

  @override
  Dim incrementIndex(Dim index) {
    if (index.dims != dims) {
      throw ArgumentError('Invalid index');
    }
    List<int> sizes = index.toList();
    for (int i = dims - 1; i >= 0; i--) {
      sizes[i]++;
      if (sizes[i] < this[i]) {
        break;
      }
      if (i > 0) {
        sizes[i] = 0;
      } else {
        sizes = toList();
      }
    }
    if (index is Dim2) {
      return Dim2(sizes[0], sizes[1]);
    } else if (index is Dim3) {
      return Dim3(sizes[0], sizes[1], sizes[2]);
    }
    return Dim(sizes);
  }

  @override
  Dim reshape(Dim newSize) {
    if (nel == 0) {
      throw StateError('Cannot reshape empty tensor');
    }
    if (nel != newSize.nel) {
      throw ArgumentError(
          'Cannot reshape. Number of elements must remain same');
    }
    return newSize;
  }

  @override
  Dim reshapeDims(int dims) {
    if (dims < this.dims) {
      throw ArgumentError('Cannot shrink dimensions');
    }
    if (dims == this.dims) {
      return this;
    }
    return Dim([...1.repeat(dims - this.dims), ...asList]);
  }

  @override
  List<int> toJson() => toList();
}

class _DimImpl with DimMixin implements Dim {
  final List<int> _sizes;

  _DimImpl(this._sizes) {
    if (_sizes.isEmpty) {
      _sizes.add(1);
    }
  }

  @override
  int get dims => _sizes.length;

  @override
  int operator [](int index) => _sizes[index];

  @override
  int get nel => _sizes.reduce((a, b) => a * b);

  @override
  int get rows {
    if (dims < 2) {
      return 1;
    }
    return _sizes[dims - 2];
  }

  @override
  int get cols {
    if (dims < 1) {
      throw StateError('Not enough dimensions');
    }
    return _sizes[dims - 1];
  }

  @override
  int get channels {
    if (dims < 3) {
      return 1;
    }
    return _sizes[dims - 3];
  }

  @override
  int get batch {
    if (dims < 4) {
      return 1;
    }
    return _sizes[dims - 4];
  }

  @override
  int get numMatrices {
    if (dims < 2) {
      return 1;
    }
    return asList.take(dims - 2).prod;
  }

  @override
  late final Dim strides = () {
    final strides = List<int>.filled(dims, 1);
    for (int i = dims - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * this[i + 1];
    }
    return Dim(strides);
  }();

  @override
  Dim2 to2D() => Dim2(rows, cols);

  Dim3 to3D() => Dim3(channels, rows, cols);

  @override
  late final asList = UnmodifiableListView(_sizes);

  @override
  List<int> toList() => List.from(_sizes);

  @override
  String toString() => 'Dim(${_sizes.join(', ')})';
}

class Dim2 with DimMixin implements Dim {
  @override
  final int rows;
  @override
  final int cols;

  const Dim2(this.rows, this.cols);

  Dim2 operator +(/* Size2D | int */ other) {
    if (other is int) {
      return Dim2(rows + other, cols + other);
    } else if (other is Dim2) {
      return Dim2(rows + other.rows, cols + other.cols);
    }
    throw ArgumentError('Invalid type');
  }

  Dim2 operator -(/* Size2D | int */ other) {
    if (other is int) {
      return Dim2(rows - other, cols - other);
    } else if (other is Dim2) {
      return Dim2(rows - other.rows, cols - other.cols);
    }
    throw ArgumentError('Invalid type');
  }

  Dim2 operator *(/* int | Size2D */ other) {
    if (other is int) {
      return Dim2(rows * other, cols * other);
    } else if (other is Dim2) {
      return Dim2(rows * other.rows, cols * other.cols);
    }
    throw ArgumentError('Invalid type');
  }

  Dim2 operator ~/(/* Size2D | int */ scalar) {
    if (scalar is int) {
      return Dim2(rows ~/ scalar, cols ~/ scalar);
    } else if (scalar is Dim2) {
      return Dim2(rows ~/ scalar.rows, cols ~/ scalar.cols);
    }
    throw ArgumentError('Invalid type');
  }

  @override
  int operator [](int index) {
    if (index == 0) {
      return rows;
    } else if (index == 1) {
      return cols;
    }
    throw RangeError('Index out of range');
  }

  @override
  int get dims => 2;

  @override
  int get nel => rows * cols;

  @override
  int get channels => 1;

  @override
  int get batch => 1;

  @override
  int get numMatrices => 1;

  @override
  Dim2 to2D() => this;

  Dim3 to3D(int channels) => Dim3(channels, rows, cols);

  @override
  Dim get strides => Dim([cols, 1]);

  @override
  UnmodifiableListView<int> get asList => UnmodifiableListView([rows, cols]);

  @override
  List<int> toList() => [rows, cols];

  @override
  String toString() => 'Dim2($rows, $cols)';
}

class Dim3 with DimMixin implements Dim {
  @override
  final int channels;
  @override
  final int rows;
  @override
  final int cols;

  const Dim3(this.channels, this.rows, this.cols);

  Dim3 operator +(/* Size3D | int */ other) {
    if (other is int) {
      return Dim3(channels + other, rows + other, cols + other);
    } else if (other is Dim3) {
      return Dim3(
          channels + other.channels, rows + other.rows, cols + other.cols);
    }
    throw ArgumentError('Invalid type');
  }

  Dim3 operator -(/* Size3D | int */ other) {
    if (other is int) {
      return Dim3(channels - other, rows - other, cols - other);
    } else if (other is Dim3) {
      return Dim3(
          channels - other.channels, rows - other.rows, cols - other.cols);
    }
    throw ArgumentError('Invalid type');
  }

  Dim3 operator *(/* int | Size3D */ other) {
    if (other is int) {
      return Dim3(channels * other, rows * other, cols * other);
    } else if (other is Dim3) {
      return Dim3(
          channels * other.channels, rows * other.rows, cols * other.cols);
    }
    throw ArgumentError('Invalid type');
  }

  Dim3 operator ~/(/* Size3D | int */ scalar) {
    if (scalar is int) {
      return Dim3(channels ~/ scalar, rows ~/ scalar, cols ~/ scalar);
    } else if (scalar is Dim3) {
      return Dim3(channels ~/ scalar.channels, rows ~/ scalar.rows,
          cols ~/ scalar.cols);
    }
    throw ArgumentError('Invalid type');
  }

  @override
  int operator [](int index) {
    if (index == 0) {
      return channels;
    } else if (index == 1) {
      return rows;
    } else if (index == 2) {
      return cols;
    }
    throw RangeError('Index out of range');
  }

  @override
  int get dims => 3;

  @override
  int get nel => channels * rows * cols;

  @override
  int get batch => 1;

  @override
  int get numMatrices => channels;

  @override
  Dim get strides => Dim([rows * cols, cols, 1]);

  @override
  UnmodifiableListView<int> get asList =>
      UnmodifiableListView([channels, rows, cols]);

  @override
  List<int> toList() => [channels, rows, cols];

  @override
  String toString() => 'Dim3($channels, $rows, $cols)';

  @override
  Dim2 to2D() => Dim2(rows, cols);
}
