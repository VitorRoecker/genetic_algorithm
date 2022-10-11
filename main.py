import numpy as np
import ga


def main():
    equation_inputs = [15, 10, 5, 5, 8, 17]
    num_weights = 6  
    solutions_per_population = 6
    population_size = (solutions_per_population, num_weights)
    population = np.random.randint(
        low=0, high=2,
        size=population_size)

    print("População inicial:")
    print(population)
    num_generations = 5
    num_parents_crossover = 4

    for generation in range(num_generations):
        print(f"\nGeração {generation}")
        fitness = ga.fitness(equation_inputs, population)
        print("\nFitness:")
        print(fitness)
        selected_parents = ga.selection(
            population, fitness, num_parents_crossover)
        print("\nGenitores selecionados:")
        print(selected_parents)
        offspring_crossover = ga.crossover(
            selected_parents, (
                solutions_per_population - num_parents_crossover,
                num_weights
            )
        )
        print("\nFilhos gerados por crossover:")
        print(offspring_crossover)
        offspring_mutation = ga.mutation(offspring_crossover)
        print("\nFilhos pós mutação:")
        print(offspring_mutation)
        population[0:selected_parents.shape[0], :] = selected_parents
        population[selected_parents.shape[0]:, :] = offspring_mutation
        print("\nNova população:")
        print(population)
        print("Melhor resultado: ", np.max(
            ga.fitness(equation_inputs, population)))
    fitness = ga.fitness(equation_inputs, population)
    best_fit_idx = np.where(fitness == np.max(fitness))
    print("Melhor resultado: ", population[best_fit_idx, :])
    print("Fitness do melhor: ", fitness[best_fit_idx])


if __name__ == "__main__":
    main()
