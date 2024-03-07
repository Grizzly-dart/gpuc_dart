abstract class Resource {
  Iterable<Context> get contexts;

  void addContext(Context context);

  void removeContext(Context context);

  void release();
}

class Context {
  final List<Resource> _resources = [];
  final List<Resource> _onError = [];

  void add(Resource resource) {
    _resources.add(resource);
    resource.addContext(this);
  }

  void releaseOnErr(Resource resource) {
    _onError.add(resource);
  }

  void release({bool isError = false}) {
    if (isError) {
      for (final resource in _onError) {
        resource.release();
      }
      _onError.clear();
    }

    for (final resources in _resources) {
      resources.removeContext(this);
      if (resources.contexts.isEmpty) {
        resources.release();
      }
    }
    _resources.clear();
  }

  Context? _child;

  Context child() {
    _child?.release();
    return _child = Context();
  }

  void releaseChild() {
    _child?.release();
    _child = null;
  }

// TODO also provide information about which devices to use

// TODO should this also contain stream?
}

class PContext {
  final Context _parent;

  PContext(this._parent);

  void add(Resource resource) {
    _parent.add(resource);
  }
}
