"""Base protocol, registry, and composite loss for the Chronofy loss suite.

TemporalLoss: ABC that all loss functions must implement. Inherits nn.Module.
LossRegistry: Plugin system for discovering and instantiating losses.
CompositeLoss: Weighted ensemble of multiple losses with dynamic add/remove.

Design principles:
  - Every loss is an nn.Module with a name property and kwargs-based forward().
  - The registry allows users to register custom losses without modifying package code.
  - CompositeLoss enables arbitrary weighted combinations, inspectable per-component.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Type

import torch
import torch.nn as nn


class TemporalLoss(nn.Module):
    """Abstract base for all Chronofy loss functions.

    Every loss must:
      1. Subclass TemporalLoss (which is also an nn.Module).
      2. Implement ``name`` — a unique string identifier (as a property).
      3. Override ``forward(**kwargs)`` — return a scalar loss tensor.

    Using **kwargs in forward() allows different losses to accept different
    inputs (embeddings, timestamps, anchors, etc.) while sharing a common
    interface for composition.

    Enforcement: ``__init_subclass__`` validates at class definition time
    that concrete subclasses provide both ``name`` and a non-inherited
    ``forward``. This is necessary because nn.Module provides a concrete
    ``forward()`` that raises NotImplementedError, which prevents Python's
    ABC mechanism from detecting missing implementations.

    Example::

        class MyLoss(TemporalLoss):
            @property
            def name(self) -> str:
                return "my_loss"

            def forward(self, *, embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
                return embeddings.norm(dim=1).mean()
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip enforcement for intermediate abstract classes
        if _is_abstract(cls):
            return
        # Check that 'name' is implemented as a property (not inherited from TemporalLoss)
        if not _has_own_property(cls, "name"):
            raise TypeError(
                f"Can't define concrete class {cls.__name__} without "
                f"implementing the 'name' property"
            )
        # Check that 'forward' is overridden (not inherited from nn.Module/TemporalLoss)
        if "forward" not in cls.__dict__:
            raise TypeError(
                f"Can't define concrete class {cls.__name__} without "
                f"implementing forward()"
            )

    @property
    def name(self) -> str:
        """Unique string identifier for this loss."""
        raise NotImplementedError

    def forward(self, **kwargs: Any) -> torch.Tensor:
        """Compute the loss. Must return a scalar tensor.

        All inputs are passed as keyword arguments so that CompositeLoss
        can route the right kwargs to each component.
        """
        raise NotImplementedError


def _is_abstract(cls: type) -> bool:
    """Check if a class is intended to be abstract (has unimplemented methods)."""
    # A class is abstract if it has abstractmethod markers or
    # if forward/name are not in its own __dict__
    return getattr(cls, "__abstractmethods__", None) is not None and len(cls.__abstractmethods__) > 0


def _has_own_property(cls: type, name: str) -> bool:
    """Check if cls defines a property (not inherited) for the given name."""
    for klass in cls.__mro__:
        if klass is TemporalLoss:
            return False  # reached base without finding it
        if name in klass.__dict__:
            return True
    return False


class LossRegistry:
    """Plugin registry for temporal loss functions.

    Allows registration, discovery, and instantiation of loss classes.
    Users can register custom losses to extend the suite without modifying
    package code.

    Example::

        registry = LossRegistry()
        registry.register(MyCustomLoss)
        loss = registry.create("my_custom_loss", temperature=0.05)
    """

    def __init__(self) -> None:
        self._registry: dict[str, Type[TemporalLoss]] = {}

    def register(
        self,
        loss_cls: Type[TemporalLoss],
        name: str | None = None,
    ) -> None:
        """Register a loss class.

        Args:
            loss_cls: The loss class to register. Must subclass TemporalLoss.
            name: Override name. If None, instantiates the class to get its name.

        Raises:
            TypeError: If loss_cls does not subclass TemporalLoss.
            ValueError: If the name is already registered.
        """
        if not (isinstance(loss_cls, type) and issubclass(loss_cls, TemporalLoss)):
            raise TypeError(
                f"Expected a TemporalLoss subclass, got {loss_cls}"
            )

        if name is None:
            # Instantiate with no args to read the name property
            try:
                instance = loss_cls()
                name = instance.name
            except TypeError:
                raise TypeError(
                    f"Cannot auto-detect name for {loss_cls.__name__}: "
                    f"constructor requires arguments. Pass name= explicitly."
                )

        if name in self._registry:
            raise ValueError(
                f"Loss '{name}' is already registered. "
                f"Use a different name or unregister first."
            )
        self._registry[name] = loss_cls

    def create(self, name: str, **kwargs: Any) -> TemporalLoss:
        """Instantiate a registered loss by name.

        Args:
            name: The registered name of the loss.
            **kwargs: Arguments forwarded to the loss constructor.

        Returns:
            A new instance of the loss.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._registry:
            raise KeyError(
                f"Loss '{name}' not found. Available: {self.available()}"
            )
        return self._registry[name](**kwargs)

    def available(self) -> list[str]:
        """Return sorted list of registered loss names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"LossRegistry({self.available()})"


class CompositeLoss(nn.Module):
    """Weighted ensemble of multiple TemporalLoss instances.

    Each component loss has a name, a weight, and receives its own kwargs
    during forward(). This enables arbitrary combinations like:

        L = 0.5 * CKA + 0.3 * TemporalContrastive + 0.1 * Smoothness

    Losses can be added, removed, and reweighted dynamically.

    Example::

        composite = CompositeLoss()
        composite.add("cka", CKALoss(), weight=0.5)
        composite.add("tc", TemporalContrastiveLoss(), weight=0.3)

        total = composite(
            cka={"X": full_emb, "Y": trunc_emb},
            tc={"embeddings": temp_emb, "timestamps": ts},
        )
    """

    def __init__(
        self,
        losses: dict[str, tuple[TemporalLoss, float]] | None = None,
    ) -> None:
        super().__init__()
        self._losses: nn.ModuleDict = nn.ModuleDict()
        self._weights: dict[str, float] = {}

        if losses:
            for name, (loss, weight) in losses.items():
                self.add(name, loss, weight)

    def add(self, name: str, loss: TemporalLoss, weight: float = 1.0) -> None:
        """Add a loss component.

        Args:
            name: Key for this component (used to route kwargs).
            loss: The loss instance.
            weight: Scalar weight for this component.
        """
        self._losses[name] = loss
        self._weights[name] = weight

    def remove(self, name: str) -> None:
        """Remove a loss component by name."""
        del self._losses[name]
        del self._weights[name]

    def update_weight(self, name: str, weight: float) -> None:
        """Update the weight of an existing component."""
        if name not in self._weights:
            raise KeyError(f"Loss '{name}' not found. Available: {self.loss_names}")
        self._weights[name] = weight

    @property
    def loss_names(self) -> list[str]:
        """Names of all registered components."""
        return list(self._losses.keys())

    def forward(self, **kwargs: Any) -> torch.Tensor:
        """Compute weighted sum of all component losses.

        Each component receives its kwargs by name. For example:
            composite(cka={"X": ..., "Y": ...}, tc={"embeddings": ...})
        routes {"X": ..., "Y": ...} to the "cka" component.

        Returns:
            Scalar tensor: sum of weight_i * loss_i(kwargs_i).
        """
        if not self._losses:
            return torch.tensor(0.0)

        total = torch.tensor(0.0)
        for name, loss_fn in self._losses.items():
            loss_kwargs = kwargs.get(name, {})
            component = loss_fn(**loss_kwargs)
            total = total + self._weights[name] * component

        return total

    def forward_with_components(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute losses and return individual component values.

        Returns:
            Tuple of (total_loss, {"name": unweighted_value, ...}).
        """
        if not self._losses:
            return torch.tensor(0.0), {}

        total = torch.tensor(0.0)
        components: dict[str, float] = {}
        for name, loss_fn in self._losses.items():
            loss_kwargs = kwargs.get(name, {})
            component = loss_fn(**loss_kwargs)
            components[name] = component.item()
            total = total + self._weights[name] * component

        return total, components
