import numpy as np
import pandas as pd


class Chromosome:

  def __init__(self, size, N, denominations) -> None:
    self.N = N
    self.size = size
    self.genes = np.random.randint(0, N, size)
    self.denominations = denominations

  @property
  def fitness(self):
    return 1 / (1 + np.abs(np.sum(self.genes * self.denominations) - self.N))

  def __lt__(self, __o: object) -> bool:
    return self.fitness > __o.fitness

  def __eq__(self, __o: object) -> bool:
    return self.fitness == __o.fitness

  def __gt__(self, __o: object) -> bool:
    return self.fitness < __o.fitness

  def single_point_crossover(self, chromosome):
    crossover_point = np.random.randint(1, self.size - 1)
    offspring1 = Chromosome(self.size, self.N, self.denominations)
    offspring1.genes = np.concatenate(
        (self.genes[:crossover_point], chromosome.genes[crossover_point:]))
    offspring2 = Chromosome(self.size, self.N, self.denominations)
    offspring2.genes = np.concatenate(
        (chromosome.genes[:crossover_point], self.genes[crossover_point:]))
    return offspring1, offspring2

  def mutate(self, mutation_probability):
    self.genes = np.where(
        np.random.random(self.size) < mutation_probability,
        np.random.randint(0, self.N, self.size), self.genes)


class GeneticAlgorithm:

  def __init__(self,
               population_size,
               denominations,
               N,
               selection_ratio=0.4,
               mutation_prob=0.75) -> None:
    self.population_size = population_size
    self.selection_ration = selection_ratio
    self.mutation_prob = mutation_prob
    self.chromosome_length = len(denominations)
    self.chromosomes = sorted([
        Chromosome(self.chromosome_length, N, denominations)
        for i in range(population_size)
    ])

  def crossover(self, parents):
    return parents[0].single_point_crossover(parents[1])

  def mutatation(self, offsprings, mutation_prob):
    for offspring in offsprings:
      offspring.mutate(mutation_prob)
    return offsprings

  def next_generation(self):
    n_selection = int(self.population_size * self.selection_ration)
    n_selection = (n_selection // 2) * 2
    fittest_individuals = self.chromosomes[:n_selection]

    offsprings = []
    for i in range(0, n_selection, 2):
      offsprings += self.crossover(fittest_individuals[i:i + 2])

    offsprings = self.mutatation(offsprings, self.mutation_prob)

    self.chromosomes += offsprings
    self.chromosomes = sorted(self.chromosomes)[:self.population_size]

  def fittest_chromosome(self):
    return self.chromosomes[0]

  def evolve(self, log_freq=1000):
    generations = 0
    while self.fittest_chromosome().fitness < 1:
      ga.next_generation()
      if generations % 100 == 0:
        print(
            f'Generation {generations}: Max fitness = {self.fittest_chromosome().fitness}'
        )
      generations += 1
    return self.fittest_chromosome()


if __name__ == '__main__':
  population_size = 10
  denominations = np.array([1, 2, 3, 4, 5, 7, 8])
  N = 50

  ga = GeneticAlgorithm(population_size, denominations, N)
  solution = ga.evolve()

  print('\nSolution Found')
  soln_table = pd.DataFrame(columns=['Denominations', 'Count'])
  soln_table['Denominations'] = denominations
  soln_table['Count'] = solution.genes
  print(soln_table.to_string(index=False))