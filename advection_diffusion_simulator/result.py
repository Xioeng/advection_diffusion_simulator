#!/usr/bin/env python
# encoding: utf-8

import pickle
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

from .config import AdvectionDiffusionConfig


@dataclass
class ADResult:
    """
    Results from advection-diffusion simulation.

    Parameters
    ----------
    meshgrid : Tuple[np.ndarray, np.ndarray]
        (X, Y) meshgrid coordinates
    solutions : list
        List of solution functions at each time step
    gradients : list
        List of gradient functions at each time step
    mesh : Any
        FEniCS mesh object
    config : AdvectionDiffusionConfig
        Configuration used for simulation
    time_steps : np.ndarray
        Array of time values
    """

    meshgrid: Tuple[np.ndarray, np.ndarray]
    solutions: np.ndarray
    config: AdvectionDiffusionConfig
    time_steps: np.ndarray
    mesh_mask: np.ndarray

    def __len__(self) -> int:
        """Return number of time steps."""
        return len(self.solutions)

    def get_solution_at_time(self, time_index: int) -> Callable:
        """
        Get solution function at given time index.

        Parameters
        ----------
        time_index : int
            Index of time step

        Returns
        -------
        Callable
            Function that evaluates solution at (x, y)
        """
        if time_index < 0 or time_index >= len(self.solutions):
            raise IndexError(
                f"Time index {time_index} out of range [0, {len(self.solutions) - 1}]"
            )

        return lambda x: np.array(self.solutions[time_index](x))

    def save(self, filepath: str) -> None:
        """
        Save results to file.

        Parameters
        ----------
        filepath : str
            Path to save file
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "ADResult":
        """
        Load results from file.

        Parameters
        ----------
        filepath : str
            Path to load file

        Returns
        -------
        ADResult
            Loaded result object
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)
