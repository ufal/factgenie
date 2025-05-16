from factgenie.prompting.strategies.base import *

# Automatically import all modules in the directory, so that the prompting strategies have a chance to register themselves.
import pkgutil
import importlib
import sys

for module in pkgutil.iter_modules(__path__):
    module_name = f"factgenie.prompting.strategies.{module.name}"
    if module_name not in sys.modules.keys():
        importlib.import_module(module_name)
        # print(f"Auto-importing module '{module_name}'.")
