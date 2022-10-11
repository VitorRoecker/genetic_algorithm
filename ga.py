import numpy as np
def fitness(inputs, numPop):

    result = np.sum(numPop * inputs, axis=1)

    fitness = []

    for num in result :
        if (num > 30 ) :
            num *= -9999999

        fitness.append(num)

    return np.array(fitness)

def selection(numPop, fitArray, num):
    parentsArray = np.empty((num, numPop.shape[1]))
    
    for idx in range(num):
        
        max_fitness_idx = np.where(fitArray == np.max(fitArray))
        max_fitness_idx = max_fitness_idx[0][0]
        parentsArray[idx, :] = numPop[max_fitness_idx, :]
        fitArray[max_fitness_idx] = -999999
    return parentsArray

def crossover(parents, generation_size):
    offspring = np.empty(generation_size)
    
    crossover_point = np.uint8(generation_size[1]/2)
    
    for idx in range(generation_size[0]):
        p1_idx = idx % parents.shape[0]
        p2_idx = (idx + 1) % parents.shape[0]
        
        offspring[idx, 0:crossover_point] = parents[
            p1_idx, 0:crossover_point]
        
        offspring[idx, crossover_point:] = parents[
            p2_idx, crossover_point:
        ]
    return offspring

def mutation(offspring):
    for idx in range(offspring.shape[0]):
        random_idx = np.random.randint(offspring.shape[1])
        
        offspring[idx, random_idx] = (
            abs(offspring[idx, random_idx] - 1)
        )
    return offspring
