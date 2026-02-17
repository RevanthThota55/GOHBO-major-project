"""
Orthogonal Learning Component

Enhances exploration capability using orthogonal experimental design
to generate diverse solutions and prevent premature convergence.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from itertools import product


class OrthogonalLearning:
    """
    Orthogonal Learning mechanism for enhancing optimization algorithms.

    Uses orthogonal arrays and experimental design to improve solution diversity
    and exploration of the search space.
    """

    def __init__(
        self,
        num_factors: int = 3,
        num_levels: int = 3,
        exploration_rate: float = 0.4,
        seed: Optional[int] = None
    ):
        """
        Initialize Orthogonal Learning component.

        Args:
            num_factors: Number of orthogonal factors
            num_levels: Number of levels for each factor
            exploration_rate: Balance between exploration and exploitation
            seed: Random seed for reproducibility
        """
        self.num_factors = num_factors
        self.num_levels = num_levels
        self.exploration_rate = exploration_rate

        if seed is not None:
            np.random.seed(seed)

        # Generate orthogonal array
        self.orthogonal_array = self._generate_orthogonal_array()

    def _generate_orthogonal_array(self) -> np.ndarray:
        """
        Generate an orthogonal array for experimental design.

        Returns:
            Orthogonal array
        """
        # For simplicity, using a basic orthogonal array design
        # For L9(3^4) orthogonal array
        if self.num_levels == 3 and self.num_factors <= 4:
            # Standard L9 orthogonal array
            array = np.array([
                [0, 0, 0, 0],
                [0, 1, 1, 1],
                [0, 2, 2, 2],
                [1, 0, 1, 2],
                [1, 1, 2, 0],
                [1, 2, 0, 1],
                [2, 0, 2, 1],
                [2, 1, 0, 2],
                [2, 2, 1, 0]
            ])
            return array[:, :self.num_factors]

        # For other cases, generate a balanced array
        return self._generate_balanced_array()

    def _generate_balanced_array(self) -> np.ndarray:
        """
        Generate a balanced array when standard orthogonal arrays are not available.

        Returns:
            Balanced array
        """
        # Generate all combinations up to a reasonable limit
        max_rows = min(self.num_levels ** self.num_factors, 27)
        array = []

        for i in range(max_rows):
            row = []
            val = i
            for _ in range(self.num_factors):
                row.append(val % self.num_levels)
                val //= self.num_levels
            array.append(row)

        return np.array(array)

    def generate_orthogonal_solutions(
        self,
        base_solutions: List[np.ndarray],
        bounds: Tuple[float, float],
        dimension: int
    ) -> List[np.ndarray]:
        """
        Generate new solutions using orthogonal learning.

        Args:
            base_solutions: List of base solutions
            bounds: Search space boundaries
            dimension: Problem dimension

        Returns:
            List of new orthogonal solutions
        """
        new_solutions = []

        for base in base_solutions:
            for orth_vector in self.orthogonal_array:
                new_solution = base.copy()

                # Apply orthogonal modifications
                for d in range(dimension):
                    # Select factor and level
                    factor_idx = d % self.num_factors
                    level = orth_vector[factor_idx]

                    # Calculate modification based on level
                    if level == 0:
                        # No change
                        pass
                    elif level == 1:
                        # Positive perturbation
                        perturbation = self.exploration_rate * (bounds[1] - bounds[0]) / 10
                        new_solution[d] += perturbation
                    elif level == 2:
                        # Negative perturbation
                        perturbation = self.exploration_rate * (bounds[1] - bounds[0]) / 10
                        new_solution[d] -= perturbation

                    # Ensure within bounds
                    new_solution[d] = np.clip(new_solution[d], bounds[0], bounds[1])

                new_solutions.append(new_solution)

        return new_solutions

    def crossover_with_orthogonal(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        dimension: int
    ) -> np.ndarray:
        """
        Perform crossover using orthogonal design principles.

        Args:
            parent1: First parent solution
            parent2: Second parent solution
            dimension: Problem dimension

        Returns:
            Offspring solution
        """
        offspring = np.zeros(dimension)

        # Select random row from orthogonal array
        orth_vector = self.orthogonal_array[
            np.random.randint(0, len(self.orthogonal_array))
        ]

        for d in range(dimension):
            factor_idx = d % self.num_factors
            level = orth_vector[factor_idx]

            if level == 0:
                # Take from parent1
                offspring[d] = parent1[d]
            elif level == 1:
                # Take from parent2
                offspring[d] = parent2[d]
            else:
                # Take average
                offspring[d] = (parent1[d] + parent2[d]) / 2

        return offspring

    def enhance_diversity(
        self,
        population: np.ndarray,
        bounds: Tuple[float, float]
    ) -> np.ndarray:
        """
        Enhance population diversity using orthogonal learning.

        Args:
            population: Current population
            bounds: Search space boundaries

        Returns:
            Enhanced population with improved diversity
        """
        pop_size, dimension = population.shape
        enhanced_pop = population.copy()

        # Calculate population centroid
        centroid = np.mean(population, axis=0)

        # Apply orthogonal perturbations
        for i in range(pop_size):
            if np.random.random() < self.exploration_rate:
                # Select orthogonal vector
                orth_idx = i % len(self.orthogonal_array)
                orth_vector = self.orthogonal_array[orth_idx]

                for d in range(dimension):
                    factor_idx = d % self.num_factors
                    level = orth_vector[factor_idx]

                    if level == 0:
                        # Move towards centroid
                        enhanced_pop[i, d] = 0.7 * enhanced_pop[i, d] + 0.3 * centroid[d]
                    elif level == 1:
                        # Move away from centroid
                        direction = enhanced_pop[i, d] - centroid[d]
                        enhanced_pop[i, d] += 0.3 * direction
                    else:
                        # Random perturbation
                        perturbation = np.random.normal(0, 0.1) * (bounds[1] - bounds[0])
                        enhanced_pop[i, d] += perturbation

                    # Clip to bounds
                    enhanced_pop[i, d] = np.clip(
                        enhanced_pop[i, d],
                        bounds[0],
                        bounds[1]
                    )

        return enhanced_pop

    def calculate_orthogonal_distance(
        self,
        solution1: np.ndarray,
        solution2: np.ndarray
    ) -> float:
        """
        Calculate orthogonal distance between two solutions.

        Args:
            solution1: First solution
            solution2: Second solution

        Returns:
            Orthogonal distance measure
        """
        # Project solutions onto orthogonal basis
        dimension = len(solution1)
        distance = 0.0

        for i in range(min(dimension, self.num_factors)):
            # Create orthogonal basis vector
            basis = np.zeros(dimension)
            basis[i] = 1.0

            # Project and calculate distance
            proj1 = np.dot(solution1, basis)
            proj2 = np.dot(solution2, basis)
            distance += (proj1 - proj2) ** 2

        return np.sqrt(distance)

    def apply_orthogonal_mutation(
        self,
        solution: np.ndarray,
        bounds: Tuple[float, float],
        mutation_strength: float = 0.1
    ) -> np.ndarray:
        """
        Apply orthogonal mutation to a solution.

        Args:
            solution: Solution to mutate
            bounds: Search space boundaries
            mutation_strength: Strength of mutation

        Returns:
            Mutated solution
        """
        mutated = solution.copy()
        dimension = len(solution)

        # Select random orthogonal vector
        orth_vector = self.orthogonal_array[
            np.random.randint(0, len(self.orthogonal_array))
        ]

        for d in range(dimension):
            factor_idx = d % self.num_factors
            level = orth_vector[factor_idx]

            # Apply mutation based on level
            mutation = 0.0
            if level == 1:
                mutation = mutation_strength * (bounds[1] - bounds[0])
            elif level == 2:
                mutation = -mutation_strength * (bounds[1] - bounds[0])

            mutated[d] += mutation

            # Add small random noise
            mutated[d] += np.random.normal(0, 0.01) * (bounds[1] - bounds[0])

            # Clip to bounds
            mutated[d] = np.clip(mutated[d], bounds[0], bounds[1])

        return mutated

    def get_orthogonal_neighbors(
        self,
        solution: np.ndarray,
        bounds: Tuple[float, float],
        num_neighbors: int = 9
    ) -> List[np.ndarray]:
        """
        Generate orthogonal neighbors of a solution.

        Args:
            solution: Base solution
            bounds: Search space boundaries
            num_neighbors: Number of neighbors to generate

        Returns:
            List of orthogonal neighbors
        """
        neighbors = []
        dimension = len(solution)

        for i in range(min(num_neighbors, len(self.orthogonal_array))):
            neighbor = solution.copy()
            orth_vector = self.orthogonal_array[i]

            for d in range(dimension):
                factor_idx = d % self.num_factors
                level = orth_vector[factor_idx]

                # Apply orthogonal perturbation
                step_size = 0.05 * (bounds[1] - bounds[0])

                if level == 0:
                    neighbor[d] -= step_size
                elif level == 1:
                    neighbor[d] += step_size
                # level == 2: no change

                # Clip to bounds
                neighbor[d] = np.clip(neighbor[d], bounds[0], bounds[1])

            neighbors.append(neighbor)

        return neighbors