import numpy as np
import imageio
import pygad
import matplotlib.pyplot as plt

class GA():
    def __init__(self, iterations, shape):
        self.size_1d = 40000
        self.size = 200
        self.iterations = iterations
        self.data = imageio.imread(f"{shape}.png")

    def run(self):
        chromosome = np.reshape(self.data, self.size_1d)

        def fitness_func(solution, solution_idx):
            fitness = np.sum(np.abs(chromosome-solution))
            fitness = np.sum(chromosome) - fitness
            return fitness

        self.ga_instance = pygad.GA(num_generations=self.iterations,
                               num_parents_mating=10,
                               fitness_func=fitness_func,
                               sol_per_pop=20,
                               num_genes=self.size_1d,
                               init_range_low=0.0,
                               init_range_high=1.0,
                               mutation_percent_genes=0.01,
                               mutation_type="random",
                               mutation_by_replacement=True,
                               random_mutation_min_val=0.0,
                               random_mutation_max_val=1.0)

        self.ga_instance.run()
        self.ga_instance.plot_result()

        self.show_results()

    def get_result(self, sol):
        return np.reshape(sol, (self.size, self.size))

    def show_results(self):
        sol, _, _ = self.ga_instance.best_solution()
        solution = self.get_result(sol)
        plt.imshow(solution)
        plt.show()
        # plt.imsave('sol.png', solution, (self.size, self.size))

        return solution
