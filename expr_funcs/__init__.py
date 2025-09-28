import importlib
import inspect
import pkgutil
import sys

def get_expr_funcs(module):
    return {
        name: obj
        for name, obj in vars(module).items()
        if inspect.isfunction(obj)
        and obj.__module__ == module.__name__
        and not name.startswith("_")
    }

expr_funcs_dict = {}

for _, modname, ispkg in pkgutil.iter_modules(__path__):
    if ispkg:
        continue
    module = importlib.import_module(f"{__name__}.{modname}")
    expr_funcs_dict.update(get_expr_funcs(module))