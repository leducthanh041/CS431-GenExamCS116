"""
demo_config.py — Wrapper Config for deploy_web demo
=====================================================
Đặt EXP_NAME = demo_<timestamp> để mỗi lần chạy demo tạo thư mục riêng,
KHÔNG ảnh hưởng config gốc trong src/common.py.
"""

import sys as _sys
from pathlib import Path as _Path

# Point sys.path to CS431MCQGen/src so 'from common import ...' works
_SRCDIR = str(_Path(__file__).resolve().parent.parent / "src")
if _SRCDIR not in _sys.path:
    _sys.path.insert(0, _SRCDIR)

from common import Config as _OrigConfig

# ─── Override EXP_NAME so each demo run goes to its own output dir ───
import datetime
DEMO_EXP_NAME = f"demo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"


class DemoConfig:
    """Proxy config: forwards everything from Config, overrides EXP_NAME."""

    def __init__(self):
        self._orig = _OrigConfig()

    def __getattr__(self, name):
        return getattr(self._orig, name)

    @property
    def EXP_NAME(self):
        return DEMO_EXP_NAME

    @property
    def OUTPUT_DIR(self):
        return self._orig.PROJECT_ROOT / "output" / DEMO_EXP_NAME

    def makedirs(self):
        """Create demo-specific output directories."""
        self._orig.makedirs()

    def set_exp_name(self, name: str):
        """Allow Streamlit to set a custom experiment name."""
        global DEMO_EXP_NAME
        DEMO_EXP_NAME = name


demo_cfg = DemoConfig()
