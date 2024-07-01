from typing import Any, Callable, Dict

__version__: str

def get_info() -> Dict[str, Any]: ...

# Lazily loaded submodules
align: Any
core: Any
data: Any
denoise: Any
direction: Any
io: Any
nn: Any
reconst: Any
segment: Any
sims: Any
stats: Any
tracking: Any
utils: Any
viz: Any
workflows: Any
tests: Any
testing: Any

# Special attributes set by lazy_loader
__getattr__: Callable[[str], Any]
__dir__: Callable[[], list[str]]
__all__: list[str]