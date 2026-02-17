"""
Grey Wolf Optimizer (GWO) Implementation

The GWO algorithm mimics the leadership hierarchy and hunting mechanism of grey wolves.
It uses three best solutions (alpha, beta, delta) to guide the search process.
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any


class GreyWolfOptimizer:
    """
    Grey Wolf Optimizer for continuous optimization problems.

    The algorithm maintains a population of wolves and uses the three best wolves
    (alpha, beta, delta) to update the positions of other wolves in the search space.
    """

    def __init__(
        self,
        objective_function: Callable,
        bounds: Tuple[float, float],
        population_size: int = 20,
        max_iterations: int = 100,
        a_initial: float = 2.0,
        a_final: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize Grey Wolf Optimizer.

        Args:
            objective_function: Function to minimize/maximize
            bounds: Search space boundaries (min, max)
            population_size: Number of wolves in the pack
            max_iterations: Maximum number of iterations
            a_initial: Initial value of parameter 'a'
            a_final: Final value of parameter 'a'
            seed: Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.a_initial = a_initial
        self.a_final = a_final

        if seed is not None:
            np.random.seed(seed)

        # Initialize wolf positions and fitness
        self.positions = None
        self.fitness = None

        # Best solutions (alpha, beta, delta wolves)
        self.alpha_position = None
        self.alpha_fitness = float('inf')
        self.beta_position = None
        self.beta_fitness = float('inf')
        self.delta_position = None
        self.delta_fitness = float('inf')

        # History for tracking convergence
        self.convergence_history = []

    def initialize_population(self, dimension: int = 1):
        """
        Initialize wolf population randomly within bounds.

        Args:
            dimension: Dimensionality of the search space
        """
        self.dimension = dimension
        self.positions = np.random.uniform(
            self.bounds[0],
            self.bounds[1],
            (self.population_size, dimension)
        )
        self.fitness = np.full(self.population_size, float('inf'))

    def evaluate_fitness(self, position: np.ndarray) -> float:
        """
        Evaluate fitness of a position.

        Args:
            position: Position in search space

        Returns:
            Fitness value
        """
        return self.objective_function(position)

    def update_leaders(self):
        """
        Update alpha, beta, and delta wolves based on fitness values.
        """
        for i in range(self.population_size):
            fitness = self.fitness[i]

            # Update alpha wolf (best solution)
            if fitness < self.alpha_fitness:
                self.delta_fitness = self.beta_fitness
                self.delta_position = self.beta_position.copy() if self.beta_position is not None else None

                self.beta_fitness = self.alpha_fitness
                self.beta_position = self.alpha_position.copy() if self.alpha_position is not None else None

                self.alpha_fitness = fitness
                self.alpha_position = self.positions[i].copy()

            # Update beta wolf (second best)
            elif fitness < self.beta_fitness:
                self.delta_fitness = self.beta_fitness
                self.delta_position = self.beta_position.copy() if self.beta_position is not None else None

                self.beta_fitness = fitness
                self.beta_position = self.positions[i].copy()

            # Update delta wolf (third best)
            elif fitness < self.delta_fitness:
                self.delta_fitness = fitness
                self.delta_position = self.positions[i].copy()

    def update_position(self, wolf_idx: int, iteration: int):
        """
        Update position of a wolf based on alpha, beta, and delta positions.

        Args:
            wolf_idx: Index of the wolf to update
            iteration: Current iteration number
        """
        # Linear decrease of parameter 'a' from 2 to 0
        a = self.a_initial - iteration * (self.a_initial - self.a_final) / self.max_iterations

        # Calculate new position influenced by alpha, beta, and delta
        new_position = np.zeros(self.dimension)

        for d in range(self.dimension):
            # Alpha influence
            r1, r2 = np.random.random(2)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * self.alpha_position[d] - self.positions[wolf_idx, d])
            X1 = self.alpha_position[d] - A1 * D_alpha

            # Beta influence
            r1, r2 = np.random.random(2)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * self.beta_position[d] - self.positions[wolf_idx, d])
            X2 = self.beta_position[d] - A2 * D_beta

            # Delta influence
            r1, r2 = np.random.random(2)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * self.delta_position[d] - self.positions[wolf_idx, d])
            X3 = self.delta_position[d] - A3 * D_delta

            # Calculate new position as average of influences
            new_position[d] = (X1 + X2 + X3) / 3

            # Ensure position stays within bounds
            new_position[d] = np.clip(new_position[d], self.bounds[0], self.bounds[1])

        self.positions[wolf_idx] = new_position

    def optimize(self, dimension: int = 1, verbose: bool = False) -> Dict[str, Any]:
        """
        Run the Grey Wolf Optimizer.

        Args:
            dimension: Dimensionality of the search space
            verbose: If True, print progress information

        Returns:
            Dictionary containing optimization results
        """
        # Initialize population
        self.initialize_population(dimension)

        # Evaluate initial population
        for i in range(self.population_size):
            self.fitness[i] = self.evaluate_fitness(self.positions[i])

        # Update leaders
        self.update_leaders()

        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Update positions of all wolves
            for i in range(self.population_size):
                self.update_position(i, iteration)

                # Evaluate new position
                self.fitness[i] = self.evaluate_fitness(self.positions[i])

            # Update alpha, beta, and delta wolves
            self.update_leaders()

            # Store convergence history
            self.convergence_history.append(self.alpha_fitness)

            if verbose and iteration % 10 == 0:
                print(f"GWO Iteration {iteration}: Best Fitness = {self.alpha_fitness:.6f}")

        return {
            'best_position': self.alpha_position,
            'best_fitness': self.alpha_fitness,
            'convergence_history': self.convergence_history,
            'alpha': self.alpha_position,
            'beta': self.beta_position,
            'delta': self.delta_position,
            'final_population': self.positions
        }

    def get_population_diversity(self) -> float:
        """
        Calculate diversity of the current population.

        Returns:
            Population diversity measure
        """
        if self.positions is None:
            return 0.0

        # Calculate pairwise distances
        diversity = 0.0
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                diversity += np.linalg.norm(self.positions[i] - self.positions[j])

        # Normalize by number of pairs
        num_pairs = self.population_size * (self.population_size - 1) / 2
        return diversity / num_pairs if num_pairs > 0 else 0.0

    def get_top_wolves(self, n: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the top n wolves and their fitness values.

        Args:
            n: Number of top wolves to return

        Returns:
            Tuple of (positions, fitness values) for top n wolves
        """
        wolves = []
        fitness_values = []

        if self.alpha_position is not None:
            wolves.append(self.alpha_position)
            fitness_values.append(self.alpha_fitness)

        if n > 1 and self.beta_position is not None:
            wolves.append(self.beta_position)
            fitness_values.append(self.beta_fitness)

        if n > 2 and self.delta_position is not None:
            wolves.append(self.delta_position)
            fitness_values.append(self.delta_fitness)

        return np.array(wolves), np.array(fitness_values)