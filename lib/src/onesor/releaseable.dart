abstract class Resource {
  void release();
}

class Context {
  final List<Resource> _resources = [];
  final List<Resource> _onError = [];

  void add(Resource resource) {
    _resources.add(resource);
  }

  void releaseOnErr(Resource resource) {
    _onError.add(resource);
  }

  void release({bool isError = false}) {
    if (isError) {
      for (final resource in _onError) {
        try {
          resource.release();
        } catch(e) {
          // TODO
        }
      }
      _onError.clear();
    }

    for (final resources in _resources) {
      try {
        resources.release();
      } catch(e) {
        // TODO
      }
    }
    _resources.clear();
  }

/*Context? _child;

  Context child() {
    _child?.release();
    return _child = Context();
  }

  void releaseChild() {
    _child?.release();
    _child = null;
  }*/

// TODO also provide information about which devices to use

// TODO should this also contain stream?
}
