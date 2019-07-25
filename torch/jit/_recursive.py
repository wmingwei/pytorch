

class WeakScriptModuleProxy(ScriptModule):
    # TODO: [weak script refactor]
    # WeakScriptModule proxy should be deleted since its functionality is
    # subsumed by recursive scripting, and the copying code in init moved
    # to a function to create a ScriptModule from an nn.Module without
    # making a WeakScriptModuleProxy
    """
    Copies the parameters, buffers, constants, attributes, and submodules
    of an nn.Module into itself.
    """
    def __init__(self, original, stubs):
        # Guards behavior of __setattr__ and __getattr__ so ScriptModule
        # __init__ can run correctly
        self.__dict__['_initialized'] = False
        super(WeakScriptModuleProxy, self).__init__(_qualified_name=_qualified_name(type(original)))
        # Store a weak reference to the original module
        self.__dict__["_original"] = weakref.ref(original)

        constants_set = set(getattr(original, "__constants__", []))
        self.__dict__["_constants_set"] = {}

        if not hasattr(original, '_parameters'):
            raise RuntimeError("'{}' has not been initialized, did you forget to call 'super()'?"
                                .format(type(original).__name__))

        # Copy Parameters and Modules
        for name in dir(original):
            item = getattr(original, name)
            if item is None and name in original._parameters:
                # XXX: treat None value simply as module attributes instead of adding them to the parameter list
                # TODO: need to handle this more generally when non-tensor attributes added to module
                object.__setattr__(self, name, item)
            elif item is self:
                continue
            elif isinstance(item, (Parameter, Module, Attribute)):
                ScriptModule.__setattr__(self, name, item)

        # Copy buffers
        for name in original._buffers:
            if original._buffers[name] is None:
                object.__setattr__(self, name, None)
            else:
                self.register_buffer(name, original._buffers[name])

        # Constants annotated via `Final[T]` rather than being added to `__constants__`
        for name, ann in getattr(original, '__annotations__', {}).items():
            if torch._jit_internal.is_final(ann):
                constants_set.add(name)

        # Copy constants
        self.__dict__["_constants_set"] = constants_set
        for name in self.__dict__["_constants_set"]:
            if hasattr(original, name):
                if (name in original._parameters or name in original._buffers) and item is not None:
                    # for 'None' parameters/buffers, don't actually add their values if it exists
                    continue
                ScriptModule.__setattr__(self, name, getattr(original, name))

        # Copy annotations, pull types from `__annotations__` or try to infer
        # the type if possible
        class_annotations = getattr(original, '__annotations__', {})
        for name in dir(original):
            if name in ("training", "__dict__"):
                # TODO: removing this skip should let us remove the code to add training as an
                # attribute in python_sugared_value.cpp
                continue
            if hasattr(self, name):
                # Don't re-copy properties
                continue
            item = getattr(original, name)
            if name in class_annotations:
                the_type = torch.jit.annotations.ann_to_type(class_annotations[name])
            else:
                the_type = torch._C._jit_try_infer_type(item)
            if the_type is not None:
                self._c._register_attribute(name, the_type, item)

        # Copy overloads
        self.__dict__["_overloads"] = dict(getattr(original, "__overloads__", {}))

        self.__dict__["_initialized"] = True
        self.__dict__["_original_type"] = type(original)
        _create_methods_from_stubs(self, stubs)

    def __getattr__(self, attr):
        # Try to get the attribute directly, if that fails, fall back to the
        # weak module itself
        try:
            return ScriptModule.__getattr__(self, attr)
        except AttributeError as e:
            # unwrap the original
            original_module = self.__dict__["_original"]()
            if original_module and self.__dict__["_initialized"]:
                # get attr from original if it is still alive
                return getattr(original_module, attr)
            elif self.__dict__["_initialized"]:
                # original module is dead, try looking up the value on the
                # original type
                fn = getattr(self.__dict__["_original_type"], attr, None)
                if fn is not None and inspect.isroutine(fn):
                    # bind the function to this instance and return it
                    return fn.__get__(self, self.__dict__["_original_type"])
            # If it's not on this module and it wasn't on the original
            # module (or the original is dead), throw the exception
            raise e

    def __setattr__(self, attr, value):
        # Once constructed, no new properties can be set

        if not self.__dict__["_initialized"]:
            # If constructing, don't fall back to original module
            return ScriptModule.__setattr__(self, attr, value)

        if hasattr(self, attr):
            return ScriptModule.__setattr__(self, attr, value)
        else:
            raise AttributeError("Cannot set new attribute '{}' on "
                                    "weak script module once it has been "
                                    "created".format(attr))


def _convert_to_script_module(mod):
    """
    Makes a ScriptModule from an nn.Module. If `_methods` is provided,
    these methods are treated as @script_methods. If not, it defaults to
    `('forward',)`. Methods accessed in forward are scripted on demand.
    """
    if isinstance(mod, ScriptModule):
        return mod

    if isinstance(mod, (ModuleList, Sequential)):
        # Create constant versions for the iterable modules
        return _create_constant_iterable_module(mod)

    methods = ()
    if hasattr(mod, 'forward'):
        if mod.forward.__func__ == torch.nn.Module.forward:
            raise RuntimeError("No forward method was defined on {}".format(mod))
        if not _jit_internal.is_ignored_fn(mod.forward):
            methods = ('forward',)
    exported = []
    for name in dir(mod):
        item = getattr(mod, name)
        if callable(item):
            if _jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.EXPORT:
                exported.append(name)
    methods = methods + tuple(exported)

    def make_stub(method):
        func = get_function_from_type(type(mod), method)
        return script_method(func, _jit_internal.createResolutionCallbackFromClosure(func))

    stubs = list(map(make_stub, methods))
    return WeakScriptModuleProxy(mod, stubs)


def _create_method_from_fn(module, fn):
    if _jit_internal.is_ignored_fn(fn):
        return None
    if not inspect.ismethod(fn):
        return None
    stub = script_method(fn, _jit_internal.createResolutionCallbackFromClosure(fn))
    with _disable_emit_hooks():
        # We don't want to call the hooks here since the graph that is calling
        # this function is not yet complete
        _create_methods_from_stubs(module, (stub,))
    return stub


def _make_strong_submodule(field, module, parent):
    if field not in parent._modules:
        # It's not a submodule, don't do anything
        return None

    # Convert the module to a ScriptModule
    new_strong_submodule = _convert_to_script_module(module)

    # Install the ScriptModule on the python side
    parent._modules._python_modules[field] = new_strong_submodule

    return new_strong_submodule


def _try_compile_fn(fn, loc):
    if _jit_internal.is_ignored_fn(fn):
        # Don't do anything for @ignore'd functions
        return None

    if isinstance(fn, torch.nn.Module):
        # Since modules are callable pybind recognizes them as functions, but
        # don't do anything for them
        return None

    if not inspect.isfunction(fn) and not inspect.ismethod(fn):
        raise RuntimeError("`{}` is not a function. Recursive scripting only supports "
                           "Python functions or methods currently.\n"
                           "Consider manually annotating `{}` with @torch.jit.script.".format(fn, fn))

    # We don't have the actual scope where the function was defined, but we can
    # extract the necessary info from the closed over variables on the function
    # object
    rcb = _jit_internal.createResolutionCallbackFromClosure(fn)
    return torch.jit.script(fn, _rcb=rcb)


def _create_constant_iterable_module(module):
    modules = OrderedDict()

    for key, submodule in module._modules.items():
        if isinstance(submodule, (ModuleList, Sequential)):
            # Make each item in the module a constant
            modules[key] = _create_constant_iterable_module(submodule)
        else:
            modules[key] = _convert_to_script_module(submodule)

    if isinstance(module, Sequential):
        return _ConstSequential(Sequential(modules))
    elif isinstance(module, ModuleList):
        return _ConstModuleList(modules)
    else:
        raise RuntimeError("Only nn.ModuleList and nn.Sequential can be made "
                           "into constant modules, found {}".format(module))
