abstract class Dim {
  factory Dim.from(value) {
    if (value is Dim) {
      return value;
    } else if (value is Iterable<int>) {
      return Dim(value);
    } else if (value is int) {
      return Dim([value]);
    } else if (value is ({int rows, int cols})) {
      return Dim.twoD(value.rows, value.cols);
    } else if (value is ({int r, int c})) {
      return Dim.twoD(value.r, value.c);
    }
    throw ArgumentError('Invalid type');
  }

  factory Dim(Iterable<int> sizes) => _DimImpl(List.from(sizes));

  factory Dim.twoD(int rows, [int? cols]) => _DimImpl([rows, cols ?? rows]);

  int operator [](int index);

  int get dims;

  Dim2 get twoD;

  int get nel;

  int get rows;

  int get cols;

  int get channels;

  int get batch;

  List<int> toList();
}

class _DimImpl implements Dim {
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
  Dim2 get twoD => Dim2(rows: rows, cols: cols);

  @override
  List<int> toList() => List.from(_sizes);

  @override
  String toString() => 'Size(${_sizes.join(', ')})';
}

class Dim2 implements Dim {
  @override
  final int rows;
  @override
  final int cols;

  const Dim2({required this.rows, required this.cols});

  Dim2 operator +(/* Size2D | int */ other) {
    if (other is int) {
      return Dim2(rows: rows + other, cols: cols + other);
    } else if (other is Dim2) {
      return Dim2(rows: rows + other.rows, cols: cols + other.cols);
    }
    throw ArgumentError('Invalid type');
  }

  Dim2 operator -(/* Size2D | int */ other) {
    if (other is int) {
      return Dim2(rows: rows - other, cols: cols - other);
    } else if (other is Dim2) {
      return Dim2(rows: rows - other.rows, cols: cols - other.cols);
    }
    throw ArgumentError('Invalid type');
  }

  Dim2 operator *(/* int | Size2D */ other) {
    if (other is int) {
      return Dim2(rows: rows * other, cols: cols * other);
    } else if (other is Dim2) {
      return Dim2(rows: rows * other.rows, cols: cols * other.cols);
    }
    throw ArgumentError('Invalid type');
  }

  Dim2 operator ~/(/* Size2D | int */ scalar) {
    if (scalar is int) {
      return Dim2(rows: rows ~/ scalar, cols: cols ~/ scalar);
    } else if (scalar is Dim2) {
      return Dim2(rows: rows ~/ scalar.rows, cols: cols ~/ scalar.cols);
    }
    throw ArgumentError('Invalid type');
  }

  @override
  List<int> toList() => [rows, cols];

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
  Dim2 get twoD => this;
}
