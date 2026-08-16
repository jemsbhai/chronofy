"""Regression tests for the published package metadata."""

from importlib.metadata import metadata

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

import chronofy

OPTIONAL_REQUIREMENTS = {
    "jsonld-ex": (">=0.7,<0.8", {"all", "sl"}),
    "networkx": (">=3.2,<4.0", {"all", "graph", "sl"}),
    "sentence-transformers": (">=3.0,<4.0", {"all", "ml"}),
    "torch": (">=2.0,<3.0", {"all", "ml"}),
}


def test_distribution_and_module_versions_match() -> None:
    distribution = metadata("chronofy")

    assert chronofy.__version__ == distribution["Version"]


def test_optional_extras_have_conditional_requirements() -> None:
    distribution = metadata("chronofy")
    extras = set(distribution.get_all("Provides-Extra") or [])
    requirements = [
        Requirement(value) for value in distribution.get_all("Requires-Dist") or []
    ]

    assert extras == {"all", "graph", "ml", "sl"}

    optional_requirements = {
        requirement.name: requirement
        for requirement in requirements
        if requirement.name in OPTIONAL_REQUIREMENTS
    }
    assert optional_requirements.keys() == OPTIONAL_REQUIREMENTS.keys()

    for name, (specifier, expected_extras) in OPTIONAL_REQUIREMENTS.items():
        requirement = optional_requirements[name]
        assert requirement.specifier == SpecifierSet(specifier)
        assert requirement.marker is not None
        active_extras = {
            extra for extra in extras if requirement.marker.evaluate({"extra": extra})
        }
        assert active_extras == expected_extras
