import numpy as np
import time

class Deep_Evolution_Strategy:

    inputs = None

    def __init__(self, weights, weights_negative, reward_function, population_size, sigma, learning_rate):
        self.weights = weights
        self.weights_negative = weights_negative
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, weights_negative, population, population_negative):
        weights_population = []
        weights_negative_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        for index, i in enumerate(population_negative):
            jittered = self.sigma * i
            weights_negative_population.append(weights_negative[index] + jittered)
        return weights_population, weights_negative_population

    def get_weights(self):
        return self.weights

    def train(self, epoch = 100, print_every = 1):
        lasttime = time.time()
        for i in range(epoch):
            population = []
            population_negative = []
            rewards = np.zeros(self.population_size)
            rewards_negative = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
                x = []
                for w in self.weights_negative:
                    x.append(np.random.randn(*w.shape))
                population_negative.append(x)
            for k in range(self.population_size):
                weights_population, weights_negative_population = self._get_weight_from_population(self.weights, self.weights_negative,
                                                                                                   population[k], population_negative[k])
                rewards[k] = self.reward_function(weights_population, weights_negative_population)
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = w + self.learning_rate/(self.population_size * self.sigma) * np.dot(A.T, rewards).T
            if (i+1) % print_every == 0:
                print('iter %d. reward: %f' %  (i+1, self.reward_function(self.weights)))
        print('time taken to train:', time.time()-lasttime, 'seconds')
