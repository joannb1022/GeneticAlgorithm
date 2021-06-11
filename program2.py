import os
import sys
import numpy as np
import scipy.misc
import itertools
import imageio
import cv2
import functools
import operator
import random

import matplotlib.pyplot as plt


def chromosome(image):
    chromosome_array = np.reshape(image, newshape=(120000,))
    return chromosome_array

def selection(population, target):

    result = np.zeros(population.shape[0])

    for i in range(population.shape[0]):

        #srednia roznicy miedzy docelowym zdjeciem a chromosomem i
        res = np.mean(np.abs(target - population[i,:]))
        res = np.sum(target) - res
        result[i] = res
    # best_indv = np.max(result)
    return result

def mutation(new_generation, parents, mutation_prob):

    size = new_generation.shape[1]
    num_mutation = int(size * mutation_prob/100)  #mutation_prob w %

    #mutujemy tylko dzieci, czyli u nas parents to liczba pierwszych chromosomow - rodzice, potem potomstwo
    for idx in range(parents, new_generation.shape[0]):
        rand_idx = np.uint32(np.random.random(size=np.uint32(num_mutation))*size)  #indeksy do podmienienia
        new_values = np.uint8(np.random.random(num_mutation) * 256)  #nowe wartosci losowe w zakresie wartosci pixeli
        new_generation[idx, rand_idx] = new_values
    return new_generation

#result to jest tuple z najlepszymi wynikami (z niej biore indeksy najlepszych rodzicow)
def reproduction(population, result, parents, img_size):


    #tablica indeskow najlepszych rodzicow
    mating = np.zeros(parents)

    for p in range(parents):
        #szukam indeksu najlepszszego i wpsiuje do tablicy mating
        # indx = np.argmin(result)
        indx = np.argmax(result)
        mating[p] = indx
        result[indx] = -1

    #powstala tablica mating to tablica najlepszych osobnikow (ich indeksow) w kolejnosci malejacej
    new_population = np.zeros((population.shape[0], 120000), dtype=np.uint8)

    #najpierw dodaje tych dobrych
    for i in range(parents):
        new_population[i] = population[int(mating[i])]

    offspring = population.shape[0] - parents  # liczba potomstwa

    pairs = list(itertools.permutations(range(0, parents), 2))  # wszystkie mozliwe pary rodzicow

    num = len(pairs)
    chosen_pairs_idx = random.sample(range(num), offspring)

    comb_idx = parents
    for comb in range(len(chosen_pairs_idx)):

        selected_comb_idx = chosen_pairs_idx[comb]
        selected_comb = pairs[selected_comb_idx]

        half_size = np.int32(new_population.shape[1]/2)
        new_population[comb_idx+comb, 0:half_size] = population[int(mating[selected_comb[0]]), 0:half_size]
        new_population[comb_idx+comb, half_size:] =  population[int(mating[selected_comb[1]]), half_size:]

    return new_population


def create_population(img_size, population_size, chromosome  = []):
    init_population = np.empty(shape=(population_size, functools.reduce(operator.mul, img_size )), dtype=np.uint8)
    for indv_num in range(population_size):
        if indv_num == 1 and len(chromosome) > 0:
            init_population[indv_num, :] = chromosome
        else:
            init_population[indv_num, :] = np.random.random(functools.reduce(operator.mul, img_size))*256

    return init_population

def show_best(result, population, img_size):
    # best = np.argmin(result[0])
    best = np.argmax(result[0])

    img_arr = np.reshape(a=population[best], newshape=img_size)
    return img_arr

def run(iterations, population, mutation_prob, target, parents, img_size):
    for i in range(iterations):
        print("Iteracja \n", i)
        #wartosci dla poszczegolnych osobnikow (jak dobrze pasuja)
        result = selection(population, target)
        print(result)
        new_generation = reproduction(population, result, parents, img_size)
        population = mutation(new_generation, parents, mutation_prob)

    image = show_best(result, population, img_size)
    return image



size = 200
population_size = 10
parents = 4
mutation_prob = 0.01
iterations = 20000
pear = cv2.imread("pear.jpg")
pear = cv2.resize(pear, (size, size))
pear_chromosome = chromosome(pear)


apple = cv2.imread("apple1.jpg")
apple = cv2.resize(apple, (size, size))
apple_chromosome = chromosome(apple)
# population = create_population(pear.shape, population_size)
population = create_population(pear.shape, population_size, chromosome = apple_chromosome)

image = run(iterations, population, mutation_prob, pear_chromosome, parents, pear.shape)
plt.imsave("im.png", image)
