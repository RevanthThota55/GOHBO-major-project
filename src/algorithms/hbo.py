"""
Heap-Based Optimizer (HBO) Implementation

HBO uses a heap data structure to efficiently manage and update solutions,
maintaining the best solutions at the top of the heap for quick access.
"""

import heapq
import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any, List


class Solution:
    """
    Represents a solution in the heap with position and fitness.
    """

    def __init__(self, position: np.ndarray, fitness: float):
        self.position = position.copy()
        self.fitness = fitness

    def __lt__(self, other):
        """For min-heap: smaller fitness is better"""
        return self.fitness < other.fitness

    def __repr__(self):
        return f"Solution(fitness={self.fitness:.6f})"


class HeapBasedOptimizer:
    """
    Heap-Based Optimizer for continuous optimization problems.

    Uses a heap structure to maintain and efficiently access the best solutions,
    with local search and replacement strategies.
    """

    def __init__(
        self,
        objective_function: Callable,
        bounds: Tuple[float, float],
        heap_size: int = 10,
        population_size: int = 30,
        max_iterations: int = 100,
        replacement_rate: float = 0.2,
        local_search_prob: float = 0.3,
        neighborhood_size: int = 5,
        seed: Optional[int] = None
    ):
        """
        Initialize Heap-Based Optimizer.

        Args:
            objective_function: Function to minimize
            bounds: Search space boundaries (min, max)
            heap_size: Size of the heap structure
            population_size: Total population size
            max_iterations: Maximum number of iterations
            replacement_rate: Rate of solution replacement
            local_search_prob: Probability of performing local search
            neighborhood_size: Size of neighborhood for local search
            seed: Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.heap_size = min(heap_size, population_size)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.replacement_rate = replacement_rate
        self.local_search_prob = local_search_prob
        self.neighborhood_size = neighborhood_size

        if seed is not None:
            np.random.seed(seed)

        # Heap structure for best solutions
        self.heap = []

        # Rest of population
        self.population = []

        # Best solution found
        self.best_solution = None
        self.best_fitness = float('inf')

        # Convergence history
        self.convergence_history = []

    def initialize_population(self, dimension: int = 1):
        """
        Initialize population and heap structure.

        Args:
            dimension: Dimensionality of the search space
        """
        self.dimension = dimension

        # Generate initial population
        all_solutions = []
        for _ in range(self.population_size):
            position = np.random.uniform(
                self.bounds[0],
                self.bounds[1],
                dimension
            )
            fitness = self.objective_function(position)
            all_solutions.append(Solution(position, fitness))

        # Sort solutions by fitness
        all_solutions.sort()

        # Fill heap with best solutions
        self.heap = all_solutions[:self.heap_size]
        heapq.heapify(self.heap)

        # Rest go to general population
        self.population = all_solutions[self.heap_size:]

        # Update best solution
        self.best_solution = self.heap[0].position.copy()
        self.best_fitness = self.heap[0].fitness

    def local_search(self, solution: Solution) -> Solution:
        """
        Perform local search around a solution.

        Args:
            solution: Current solution

        Returns:
            Improved solution (or original if no improvement)
        """
        best_local = solution

        for _ in range(self.neighborhood_size):
            # Generate neighbor
            perturbation = np.random.normal(0, 0.1, self.dimension)
            neighbor_position = solution.position + perturbation

            # Clip to bounds
            neighbor_position = np.clip(
                neighbor_position,
                self.bounds[0],
                self.bounds[1]
            )

            # Evaluate neighbor
            neighbor_fitness = self.objective_function(neighbor_position)

            # Update if better
            if neighbor_fitness < best_local.fitness:
                best_local = Solution(neighbor_position, neighbor_fitness)

        return best_local

    def update_heap(self, new_solution: Solution):
        """
        Update heap with a new solution if it's better than the worst in heap.

        Args:
            new_solution: New solution to potentially add to heap
        """
        if len(self.heap) < self.heap_size:
            heapq.heappush(self.heap, new_solution)
        elif new_solution.fitness < max(self.heap, key=lambda x: x.fitness).fitness:
            # Remove worst solution from heap
            self.heap.sort()
            self.heap.pop()
            # Add new solution
            self.heap.append(new_solution)
            heapq.heapify(self.heap)

    def generate_new_solution(self) -> Solution:
        """
        Generate a new solution based on heap solutions.

        Returns:
            New solution
        """
        # Select random solutions from heap
        if len(self.heap) >= 3:
            selected = np.random.choice(len(self.heap), 3, replace=False)
            s1, s2, s3 = [self.heap[i] for i in selected]

            # Differential evolution style generation
            F = np.random.uniform(0.4, 1.0)  # Scaling factor
            new_position = s1.position + F * (s2.position - s3.position)
        else:
            # Random generation if heap is too small
            new_position = np.random.uniform(
                self.bounds[0],
                self.bounds[1],
                self.dimension
            )

        # Clip to bounds
        new_position = np.clip(new_position, self.bounds[0], self.bounds[1])

        # Evaluate
        new_fitness = self.objective_function(new_position)

        return Solution(new_position, new_fitness)

    def replace_worst_solutions(self):
        """
        Replace worst solutions in population with new ones.
        """
        num_replace = int(self.replacement_rate * len(self.population))

        if num_replace > 0:
            # Sort population
            self.population.sort(reverse=True)  # Worst first

            # Replace worst solutions
            for i in range(num_replace):
                new_solution = self.generate_new_solution()

                # Apply local search with probability
                if np.random.random() < self.local_search_prob:
                    new_solution = self.local_search(new_solution)

                self.population[i] = new_solution

                # Try to update heap
                self.update_heap(new_solution)

    def optimize(self, dimension: int = 1, verbose: bool = False) -> Dict[str, Any]:
        """
        Run the Heap-Based Optimizer.

        Args:
            dimension: Dimensionality of the search space
            verbose: If True, print progress information

        Returns:
            Dictionary containing optimization results
        """
        # Initialize
        self.initialize_population(dimension)

        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Generate new solutions based on heap
            new_solutions = []
            for _ in range(self.population_size - self.heap_size):
                new_solution = self.generate_new_solution()

                # Local search with probability
                if np.random.random() < self.local_search_prob:
                    new_solution = self.local_search(new_solution)

                new_solutions.append(new_solution)

            # Update population
            self.population = new_solutions

            # Update heap with best from new population
            for solution in self.population:
                self.update_heap(solution)

            # Replace worst solutions
            self.replace_worst_solutions()

            # Update best solution
            heap_best = min(self.heap, key=lambda x: x.fitness)
            if heap_best.fitness < self.best_fitness:
                self.best_fitness = heap_best.fitness
                self.best_solution = heap_best.position.copy()

            # Store convergence history
            self.convergence_history.append(self.best_fitness)

            if verbose and iteration % 10 == 0:
                print(f"HBO Iteration {iteration}: Best Fitness = {self.best_fitness:.6f}")

        return {
            'best_position': self.best_solution,
            'best_fitness': self.best_fitness,
            'convergence_history': self.convergence_history,
            'heap_solutions': [(s.position, s.fitness) for s in self.heap],
            'final_population': [(s.position, s.fitness) for s in self.population]
        }

    def get_heap_diversity(self) -> float:
        """
        Calculate diversity of solutions in the heap.

        Returns:
            Heap diversity measure
        """
        if len(self.heap) < 2:
            return 0.0

        diversity = 0.0
        count = 0

        for i in range(len(self.heap)):
            for j in range(i + 1, len(self.heap)):
                diversity += np.linalg.norm(
                    self.heap[i].position - self.heap[j].position
                )
                count += 1

        return diversity / count if count > 0 else 0.0

    def get_top_solutions(self, n: int = 5) -> List[Tuple[np.ndarray, float]]:
        """
        Get the top n solutions from the heap.

        Args:
            n: Number of top solutions to return

        Returns:
            List of (position, fitness) tuples
        """
        sorted_heap = sorted(self.heap)
        return [(s.position, s.fitness) for s in sorted_heap[:min(n, len(sorted_heap))]]