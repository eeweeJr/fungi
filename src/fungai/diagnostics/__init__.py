"""Instrumentation for the observation model: activations, attention, gradients,
drift, part-attention, invariance checks and health alarms."""

from fungai.diagnostics.probes import Diagnostics, parameter_groups

__all__ = ["Diagnostics", "parameter_groups"]
