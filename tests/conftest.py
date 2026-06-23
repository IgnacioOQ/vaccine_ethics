"""Shared pytest fixtures and path setup for the engine tests.

Puts ``src/`` on the import path so ``from imports import *`` /
``from simulation_class import ...`` resolve the same way they do inside the
strand notebooks, and forces a non-interactive matplotlib backend so tests run
headless.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: no figure windows during tests

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
