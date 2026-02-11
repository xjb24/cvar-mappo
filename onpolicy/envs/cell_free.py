import importlib.util
import sys
import types
from pathlib import Path

CELL_FREE_DIR = Path(__file__).resolve().parent / "cell-free_env"
PACKAGE_NAME = "cell_free_env"

# Register a synthetic package so relative imports inside the env work.
if PACKAGE_NAME not in sys.modules:
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(CELL_FREE_DIR)]
    sys.modules[PACKAGE_NAME] = pkg


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config_module = _load_module(f"{PACKAGE_NAME}.config", CELL_FREE_DIR / "config.py")
env_module = _load_module(f"{PACKAGE_NAME}.cell_free_env", CELL_FREE_DIR / "cell_free_env.py")

CellFreeEnv = env_module.CellFreeEnv
MultiAgentCellFreeEnv = env_module.MultiAgentCellFreeEnv
make_env = env_module.make_env
make_multiagent_env = env_module.make_multiagent_env

OBS_VECTOR_SIZE = env_module.OBS_VECTOR_SIZE
NUM_APS = config_module.NUM_APS
NUM_USERS = config_module.NUM_USERS
NUM_SUBCARRIERS = config_module.NUM_SUBCARRIERS
TIME_STEPS = config_module.TIME_STEPS

__all__ = [
    "CellFreeEnv",
    "MultiAgentCellFreeEnv",
    "make_env",
    "make_multiagent_env",
    "NUM_APS",
    "NUM_USERS",
    "OBS_VECTOR_SIZE",
    "NUM_SUBCARRIERS",
    "TIME_STEPS",
]
