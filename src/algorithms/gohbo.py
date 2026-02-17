"""
GOHBO (Grey Wolf + Orthogonal Learning enhanced Heap-Based Optimization)

This is the main integrated algorithm that combines:
1. Grey Wolf Optimizer (GWO) for leadership-based search
2. Heap-Based Optimizer (HBO) for efficient solution management
3. Orthogonal Learning (OL) for enhanced exploration
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any, List
from .gwo import GreyWolfOptimizer
from .hbo import HeapBasedOptimizer, Solution
from .orthogonal import OrthogonalLearning


class GOHBO:
    """
    GOHBO: An advanced hybrid optimization algorithm for hyperparameter tuning.

    Combines the strengths of GWO, HBO, and Orthogonal Learning to achieve
    robust optimization performance for complex problems.
    """

    def __init__(
        self,
        objective_function: Callable,
        bounds: Tuple[float, float],
        population_size: int = 30,
        max_iterations: int = 100,
        gwo_config: Optional[Dict] = None,
        hbo_config: Optional[Dict] = None,
        orthogonal_config: Optional[Dict] = None,
        seed: Optional[int] = None,
        use_log_scale: bool = True
    ):
        """
        Initialize GOHBO algorithm.

        Args:
            objective_function: Function to minimize (e.g., validation loss)
            bounds: Search space boundaries for learning rate
            population_size: Total population size
            max_iterations: Maximum number of iterations
            gwo_config: Configuration for GWO component
            hbo_config: Configuration for HBO component
            orthogonal_config: Configuration for Orthogonal Learning
            seed: Random seed for reproducibility
            use_log_scale: Use logarithmic scale for learning rate
        """
        self.objective_function = objective_function
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.use_log_scale = use_log_scale

        if seed is not None:
            np.random.seed(seed)

        # Initialize components with default configs if not provided
        self.gwo_config = gwo_config or {
            'a_initial': 2.0,
            'a_final': 0.0,
            'alpha_weight': 0.5,
            'beta_weight': 0.3,
            'delta_weight': 0.2
        }

        self.hbo_config = hbo_config or {
            'heap_size': 10,
            'replacement_rate': 0.2,
            'local_search_prob': 0.3,
            'neighborhood_size': 5
        }

        self.orthogonal_config = orthogonal_config or {
            'num_factors': 3,
            'num_levels': 3,
            'exploration_rate': 0.4
        }

        # Initialize components
        self._initialize_components()

        # Population and fitness tracking
        self.population = None
        self.fitness = None
        self.best_position = None
        self.best_fitness = float('inf')

        # Convergence history
        self.convergence_history = []
        self.diversity_history = []

    def _initialize_components(self):
        """Initialize GWO, HBO, and Orthogonal Learning components."""
        # Grey Wolf Optimizer
        self.gwo = GreyWolfOptimizer(
            objective_function=self.objective_function,
            bounds=self._transform_bounds() if self.use_log_scale else self.bounds,
            population_size=self.population_size // 2,  # Half for GWO
            max_iterations=self.max_iterations,
            a_initial=self.gwo_config['a_initial'],
            a_final=self.gwo_config['a_final']
        )

        # Heap-Based Optimizer
        self.hbo = HeapBasedOptimizer(
            objective_function=self.objective_function,
            bounds=self._transform_bounds() if self.use_log_scale else self.bounds,
            heap_size=self.hbo_config['heap_size'],
            population_size=self.population_size // 2,  # Half for HBO
            max_iterations=self.max_iterations,
            replacement_rate=self.hbo_config['replacement_rate'],
            local_search_prob=self.hbo_config['local_search_prob'],
            neighborhood_size=self.hbo_config['neighborhood_size']
        )

        # Orthogonal Learning
        self.orthogonal = OrthogonalLearning(
            num_factors=self.orthogonal_config['num_factors'],
            num_levels=self.orthogonal_config['num_levels'],
            exploration_rate=self.orthogonal_config['exploration_rate']
        )

    def _transform_bounds(self) -> Tuple[float, float]:
        """Transform bounds to log scale if needed."""
        if self.use_log_scale:
            return (np.log10(self.bounds[0]), np.log10(self.bounds[1]))
        return self.bounds

    def _transform_to_original_scale(self, value: float) -> float:
        """Transform value back from log scale if needed."""
        if self.use_log_scale:
            return 10 ** value
        return value

    def _evaluate_fitness(self, position: np.ndarray) -> float:
        """
        Evaluate fitness of a position, handling scale transformation.

        Args:
            position: Position in search space (potentially in log scale)

        Returns:
            Fitness value
        """
        # Transform back to original scale if needed
        actual_value = self._transform_to_original_scale(position[0])
        return self.objective_function(actual_value)

    def initialize_population(self):
        """Initialize the combined population."""
        # Initialize GWO population
        self.gwo.initialize_population(dimension=1)  # 1D for learning rate
        for i in range(self.gwo.population_size):
            self.gwo.fitness[i] = self._evaluate_fitness(self.gwo.positions[i])
        self.gwo.update_leaders()

        # Initialize HBO population
        self.hbo.initialize_population(dimension=1)

        # Combine populations
        gwo_pop = self.gwo.positions
        hbo_pop = np.array([s.position for s in self.hbo.heap + self.hbo.population])

        self.population = np.vstack([gwo_pop, hbo_pop])
        self.fitness = np.zeros(len(self.population))

        # Evaluate combined population
        for i in range(len(self.population)):
            self.fitness[i] = self._evaluate_fitness(self.population[i])

        # Update best solution
        best_idx = np.argmin(self.fitness)
        self.best_position = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

    def hybrid_update(self, iteration: int):
        """
        Perform hybrid update using all three components.

        Args:
            iteration: Current iteration number
        """
        # Get bounds for operations
        bounds = self._transform_bounds() if self.use_log_scale else self.bounds

        # Step 1: GWO update for first half of population
        for i in range(self.gwo.population_size):
            self.gwo.update_position(i, iteration)
            self.gwo.fitness[i] = self._evaluate_fitness(self.gwo.positions[i])
        self.gwo.update_leaders()

        # Step 2: HBO update for second half
        # Generate new solutions based on heap
        new_hbo_solutions = []
        for _ in range(self.hbo.population_size):
            new_solution = self.hbo.generate_new_solution()

            # Apply local search with probability
            if np.random.random() < self.hbo.local_search_prob:
                new_solution = self.hbo.local_search(new_solution)

            new_hbo_solutions.append(new_solution)

            # Update heap
            self.hbo.update_heap(new_solution)

        # Step 3: Apply Orthogonal Learning for diversity enhancement
        # Get top solutions from both components
        gwo_top, _ = self.gwo.get_top_wolves(3)
        hbo_top = [s[0] for s in self.hbo.get_top_solutions(3)]
        top_solutions = list(gwo_top) + hbo_top

        # Generate orthogonal solutions
        orthogonal_solutions = self.orthogonal.generate_orthogonal_solutions(
            top_solutions,
            bounds,
            dimension=1
        )

        # Step 4: Information exchange between GWO and HBO
        # Best GWO wolves influence HBO
        if self.gwo.alpha_position is not None:
            alpha_solution = Solution(
                self.gwo.alpha_position,
                self.gwo.alpha_fitness
            )
            self.hbo.update_heap(alpha_solution)

        # Best HBO solutions influence GWO
        if len(self.hbo.heap) > 0:
            best_hbo = min(self.hbo.heap, key=lambda x: x.fitness)
            # Replace worst wolf with best HBO solution
            worst_wolf_idx = np.argmax(self.gwo.fitness)
            self.gwo.positions[worst_wolf_idx] = best_hbo.position.copy()
            self.gwo.fitness[worst_wolf_idx] = best_hbo.fitness

        # Step 5: Evaluate orthogonal solutions and update population
        for orth_sol in orthogonal_solutions[:5]:  # Use top 5 orthogonal solutions
            fitness = self._evaluate_fitness(orth_sol)

            # Replace worst solution in combined population
            worst_idx = np.argmax(self.fitness)
            if fitness < self.fitness[worst_idx]:
                self.population[worst_idx] = orth_sol
                self.fitness[worst_idx] = fitness

        # Update combined population
        self.population[:self.gwo.population_size] = self.gwo.positions
        self.population[self.gwo.population_size:] = np.array(
            [s.position for s in self.hbo.heap[:len(self.hbo.heap)//2]] +
            [s.position for s in self.hbo.population[:len(self.hbo.population)//2]]
        )[:self.hbo.population_size]

        # Re-evaluate fitness
        for i in range(len(self.population)):
            self.fitness[i] = self._evaluate_fitness(self.population[i])

    def optimize(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run the GOHBO optimization algorithm.

        Args:
            verbose: If True, print progress information

        Returns:
            Dictionary containing optimization results
        """
        # Initialize population
        self.initialize_population()

        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Perform hybrid update
            self.hybrid_update(iteration)

            # Update best solution
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_position = self.population[current_best_idx].copy()

            # Calculate and store diversity
            diversity = self.calculate_population_diversity()
            self.diversity_history.append(diversity)

            # Store convergence history
            self.convergence_history.append(self.best_fitness)

            if verbose and iteration % 10 == 0:
                actual_lr = self._transform_to_original_scale(self.best_position[0])
                print(f"GOHBO Iteration {iteration}: "
                      f"Best LR = {actual_lr:.6f}, "
                      f"Best Fitness = {self.best_fitness:.6f}, "
                      f"Diversity = {diversity:.4f}")

            # Early stopping if converged
            if len(self.convergence_history) > 10:
                recent = self.convergence_history[-10:]
                if np.std(recent) < 1e-6:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break

        # Transform best position back to original scale
        best_lr = self._transform_to_original_scale(self.best_position[0])

        return {
            'best_learning_rate': best_lr,
            'best_fitness': self.best_fitness,
            'best_position': self.best_position,
            'convergence_history': self.convergence_history,
            'diversity_history': self.diversity_history,
            'final_population': self.population,
            'final_fitness': self.fitness,
            'gwo_alpha': self.gwo.alpha_position,
            'gwo_beta': self.gwo.beta_position,
            'gwo_delta': self.gwo.delta_position,
            'hbo_heap': self.hbo.get_top_solutions(5),
            'iterations_completed': iteration + 1
        }

    def calculate_population_diversity(self) -> float:
        """
        Calculate diversity of the current population.

        Returns:
            Population diversity measure
        """
        if self.population is None or len(self.population) < 2:
            return 0.0

        # Calculate average pairwise distance
        diversity = 0.0
        count = 0

        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                diversity += np.abs(self.population[i, 0] - self.population[j, 0])
                count += 1

        return diversity / count if count > 0 else 0.0

    def get_optimization_summary(self) -> str:
        """
        Get a summary of the optimization results.

        Returns:
            String summary of optimization
        """
        if self.best_position is None:
            return "Optimization not yet performed"

        best_lr = self._transform_to_original_scale(self.best_position[0])

        summary = f"""
GOHBO Optimization Summary
========================
Best Learning Rate: {best_lr:.6f}
Best Fitness: {self.best_fitness:.6f}
Total Iterations: {len(self.convergence_history)}
Final Diversity: {self.diversity_history[-1] if self.diversity_history else 0:.4f}

Component Performance:
- GWO Alpha Fitness: {self.gwo.alpha_fitness:.6f}
- HBO Best Fitness: {self.hbo.best_fitness:.6f}

Convergence:
- Initial Fitness: {self.convergence_history[0] if self.convergence_history else 0:.6f}
- Final Fitness: {self.convergence_history[-1] if self.convergence_history else 0:.6f}
- Improvement: {(self.convergence_history[0] - self.convergence_history[-1]) if self.convergence_history else 0:.6f}
"""
        return summary