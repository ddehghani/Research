import os
import importlib

# Automatically import all *_utils.py modules in the utils package
utils_dir = os.path.dirname(__file__)
for filename in os.listdir(utils_dir):
    if filename.endswith("_utils.py"):
        module_name = filename[:-3]
        module = importlib.import_module(f".{module_name}", package=__name__)
        globals().update({
            name: getattr(module, name)
            for name in dir(module)
            if not name.startswith("_")
        })