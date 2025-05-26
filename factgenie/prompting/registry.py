import logging
from inspect import isabstract

logger = logging.getLogger("factgenie")


# Tutorial on how to use the classes below:
#
# For a simple registry that doesn't warn users when they forget to register their class:
#     register_foo = Registry(FooBaseClass, "register_foo")
# This gains you the ability to use @register_foo as a decorator above any Foo(FooBaseClass).
# To then access the registered subclasses, use:
#     register_foo.registered_subclasses
#
# If you wish users to get notified when they forget to register their class, also add:
#     @track_subclasses above FooBaseClass
#     unregistered_foo_tracker = UnregisteredTracker(FooBaseClass, [register_foo])
# Then to see a warning about newly discovered unregistered subclasses, just call:
#     unregistered_foo_tracker.warn_about_unregistered_subclasses()
#
# To supress warnings about unregistered subclass on mock classes, use the @untracked decorator.
#
# WARNING: The classes won't be found if their file is never importred!


# This creates a registry, e.g. `register_llm_gen = Registry(...)`.
# It works as a decorator because it is callable.
class Registry:
    def __init__(self, base_type, decorator_name: str):
        """
        Args:
            base_type: The type of the subclass base.
            decorator_name: The exact same as the name of the instance you are creating. E.g. `register_llm_gen = Registry(..., "register_llm_gen")`.
        """
        self.base_type = base_type
        self.decorator_name = decorator_name
        self.registered_subclasses = {}

    def _add_to_registry(self, subclass, key: str):
        assert (
            key not in self.registered_subclasses
        ), f"Multiple `{self.base_type.__name__}` subclasses are trying to register under the same name ('{key}') using @{self.decorator_name}!"
        self.registered_subclasses[key] = subclass
        # This log is currently commented out because it got called before the logger was set-up. It's also not clear whether it's beneficial. It should also probably use logger.debug(...).
        # logger.info(f"The class {subclass.__name__} has been registered using @{self.instance_name} under the name '{key}'.")

    # Decorators with arguments are actually just regular functions that return the actual decotaror taking a single argument (of the decoratee).
    def __call__(self, name: str):
        """
        Registers this class to be visible to factgenie as an LLM_GEN strategy under the selected name.
        """

        # The actual decorator just registers the class and returns it unchanged.
        def register(original_class):
            assert issubclass(
                original_class, self.base_type
            ), f"'@{self.decorator_name}' can only be used to subclasses of `{self.base_type.__name__}`."
            self._add_to_registry(original_class, name)
            return original_class

        return register


# By decorating a class with this, it gains the __init_subclass__, that will register all subclasses (unless the class is abstract or has tracking removed by `@untracked`).
def track_subclasses(original_class):
    old_init_subclass = getattr(original_class, "__init_subclass__", None)

    # Register all found subclasses so we can warn users when they forget to register their class.
    # https://docs.python.org/3/reference/datamodel.html#object.__init_subclass__
    @classmethod
    def init_subclass(cls, **kwargs) -> None:
        if old_init_subclass is not None:
            old_init_subclass(**kwargs)

        # Abstract strategies don't need to be registered.
        if not (isabstract(cls)):
            original_class._FOUND_SUBCLASSES.add(cls)

    # I am not sure if naming it in uppercase is correct, but it basically acts as a constant after the "compilation" of the program.
    original_class._FOUND_SUBCLASSES = set()
    original_class.__init_subclass__ = init_subclass
    return original_class


# This removes tracking from the class. Useful for mocking classes and non-abstract bases.
def untracked(original_class):
    # The classes are registered using `__init_subclass__`, which is done before this decorator is applied. Therefore we have to remove the tracker after it has been applied.
    # To find the tracker, we use `getattr`, as it finds it on the ancestor class.
    tracking = getattr(original_class, "_FOUND_SUBCLASSES", None)
    if tracking is not None:
        tracking -= {original_class}
    return original_class


# This class adds the ability to warn users when they forget to register their class. The registration concept would actually work just fine without any tracking/warning. This was more of an exercise for me (- Filip Rechtorík), but the free warnings are quite nice for users.
class UnregisteredTracker:
    def __init__(self, base_type, registries: list[Registry]):
        assert (
            getattr(base_type, "_FOUND_SUBCLASSES", None) is not None
        ), f"Cannot use `{UnregisteredTracker.__name__}` on a class without the @{track_subclasses.__name__} attribute!"
        assert len(registries) > 0, f"Need to have at least one registry to use `{UnregisteredTracker.__name__}`"

        self.base_type = base_type
        self.registries = registries
        self.checked = set()

    def warn_about_unregistered_subclasses(self):
        """
        This function produces warnings for classes that don't have any registration using the provided @register_llm_gen or @register_llm_eval. Each class will be reported only once, no matter how many times this function is called.
        """
        all_subclasses = getattr(self.base_type, "_FOUND_SUBCLASSES")  # Already assertded to exist

        # We want to warn about all subclasses that were not checked yet and that are not registered.
        unregistered_classes = all_subclasses - self.checked
        for registry in self.registries:
            unregistered_classes -= set(registry.registered_subclasses.values())

        # All remaining subclasses are unregistered.
        for unregistered_class in unregistered_classes:
            logger.warning(
                f"The class `{unregistered_class.__name__}` was found but not registered. It will not be visible to factgenie! To fix this, decorate the class using one of the following decorators: {', '.join(map(lambda x: f'@{x.decorator_name}', self.registries))}, @{untracked.__name__}."
            )
            # logger.warning(f"The prompting strategy `{unregistered_class.__name__}` was found but not registered. It will not be visible to factgenie! To fix this, decorate the class by either @register_llm_gen or @register_llm_eval.")

        self.checked |= all_subclasses
