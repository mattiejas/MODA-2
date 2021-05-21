import numpy as np
import math
import time
from tqdm import tqdm

t_max = 1000
p_size = 100
stepsize = np.random.uniform(0, 1, size=(5, 45))


def generate_individual():
    # (a, b, x1...xk, y1...yk, s1...sk)
    a, b = (
        np.random.randint(low=10, high=101),
        np.random.randint(low=10, high=36),
    )
    x = np.random.randint(low=5, high=a-5+1, size=(45))
    y = np.random.randint(low=5, high=b-5+1, size=(45))
    s = np.random.randint(low=0, high=2, size=(45))
    return [a, b, x, y, s]


def init_population(amount=2):
    return [generate_individual() for _ in range(amount)]


def is_feasible(p):
    x = p[2]
    y = p[3]
    s = p[4]

    for i in range(len(p[4])):
        for j in range(i + 1, len(p[4])):
            if s[i] * s[j] * (10 - math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)) < 0:
                return False
    return True


def evaluate(p):
    # 45 - amount
    score = 0
    feasible = is_feasible(p)

    if not feasible:
        score += 45 + (100 * 35)

    return [feasible, (score + (45 - np.sum(p[4])) + (p[0] * p[1]))]


def mutate(p):
    p[0] += stepsize[0][0] * np.random.normal()
    p[1] += stepsize[1][0] * np.random.normal()
    p[2] = np.array([x + stepsize[2][i] * np.random.normal() for i, x in enumerate(p[2])])
    p[3] = np.array([x + stepsize[3][i] * np.random.normal() for i, x in enumerate(p[3])])
    p[4] = np.array([x if np.random.uniform(0, 1) > stepsize[4][i] else int(not x) for i, x in enumerate(p[4])])
    return p


if __name__ == '__main__':
    seconds = time.time()

    # init population
    t = 0
    population = init_population(p_size)

    for t in tqdm(range(t_max)):
        # generate
        idx = np.random.randint(0, p_size)
        x_new = np.copy(population[idx]).tolist()
        population = [*population, x_new]

        # evaluate
        evaluation = np.array([evaluate(x) for x in population])
        feasibility = evaluation[:, 0]
        score = evaluation[:, 1]

        # delete random candidate
        candidate = None
        infeasible_solutions = np.argwhere(feasibility == False).flatten()
        if len(infeasible_solutions) > 0:  # select random infeasible candidate
            candidate = np.random.choice(infeasible_solutions)
        else:  # select lowest
            candidate = np.argmax(score)

        del population[candidate]

        t += 1

    delta = time.time() - seconds
    print(f'done in {delta} seconds.')

# minimize the amount of remaining dough
# maximize the number of cookies

# 35 · 100/(5 · 5π) ≈ 44.563
# area of a single cookie is 25pi

# constraint
# 5 ≤ xi ≤ a − 5, 5 ≤ yi ≤ b − 5, i = 1, ..., k

# objectives
# sum of active cookies
# area of dough

# individual
