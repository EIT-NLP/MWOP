from __future__ import annotations
import importlib
import importlib.util
from functools import lru_cache
from typing import Any, Optional, Tuple

@lru_cache(maxsize=128)
def is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None

def optional_import(module_name: str, attribute: Optional[str]=None, fallback: Any=None) -> Tuple[Any, bool]:
    try:
        module = importlib.import_module(module_name)
        if attribute is not None:
            return (getattr(module, attribute), True)
        return (module, True)
    except (ImportError, AttributeError):
        return (fallback, False)

class MissingOptionalDependencyError(ImportError):

    def __init__(self, package: str, extras: Optional[str]=None, feature: Optional[str]=None):
        if extras:
            install_cmd = f'pip install lmms_eval[{extras}]'
        else:
            install_cmd = f'pip install {package}'
        feature_msg = f' for {feature}' if feature else ''
        message = f"'{package}' is required{feature_msg} but not installed. Install with: {install_cmd}"
        super().__init__(message)

def require_package(package: str, extras: Optional[str]=None, feature: Optional[str]=None) -> None:
    if not is_package_available(package):
        raise MissingOptionalDependencyError(package, extras, feature)

def make_lazy_getattr(lazy_imports: dict[str, tuple[str, str]]):

    def __getattr__(name: str) -> Any:
        if name in lazy_imports:
            module_path, attr_name = lazy_imports[name]
            module = importlib.import_module(module_path)
            return getattr(module, attr_name)
        raise AttributeError(f'module has no attribute {name!r}')
    return __getattr__
