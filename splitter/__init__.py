"""
Splitter Module

This module contains model components and utilities for determining
document boundaries using multimodal (text + image) signals.

It includes model definitions, training scripts, inference logic,
and evaluation utilities.

Classes
-------
FusionModel
    A neural network model that fuses outputs from a CNN and a language model.
"""

from .model import FusionModel
