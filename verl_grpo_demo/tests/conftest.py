"""pytest bootstrap for verl_grpo_demo tests.

Adds project root to sys.path so `import evaluation` and `import utils`
work no matter where pytest is invoked from. pytest auto-loads any
conftest.py at or above the collected tests, so this file needs no manual
registration.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)