import numpy as np
import math
import time
from tqdm import trange
from plot import plot_solution

t_max = 10000
p_size = 8
# Max step size
# a, b, x, y, s
step_size = [.5, .5, .5, .5, 0.05]
crossover_rate = 0.2
number_of_starting = 5


def generate_individual():
    # (a, b, x1...xk, y1...yk, s1...sk)
    a, b = (
        np.random.randint(low=10, high=101),
        np.random.randint(low=10, high=36),
    )
    x = np.random.randint(low=5, high=a-5+1, size=(45))
    y = np.random.randint(low=5, high=b-5+1, size=(45))
    # s = np.random.randint(low=0, high=2, size=(45))

    s = np.zeros(shape=(45))
    idxs = np.random.choice(range(45), size=number_of_starting)
    s[idxs] = 1

    return [a, b, x, y, s]


def init_population(amount=2):
    return [generate_individual() for _ in range(amount)]


def calculate_feasibility(p):
    a, b, x, y, s = p
    incorrect_overlap_penalty = 0

    for i in range(len(p[4])):
        for j in range(i + 1, len(p[4])):
            incorrect_overlap_penalty += calculate_overlap_penalty(p, i, j)

    return incorrect_overlap_penalty, not bool(incorrect_overlap_penalty)


def calculate_position_penalty(p, i):
    a, b, x, y, s = p
    penalty = 0

    if s[i] == 1 and not (5 <= x[i] <= a - 5 and 5 <= y[i] <= b - 5):
        if not (5 <= x[i] <= a - 5):
            penalty += min(abs(5 - x[i]), abs(x[i] - (a - 5))) ** 2
        if not (5 <= y[i] <= b - 5):
            penalty += min(abs(5 - y[i]), abs(y[i] - (b - 5))) ** 2

    return penalty


def calculate_overlap_penalty(p, i, j):
    a, b, x, y, s = p
    penalty = 0
    count = 0

    if s[i] * s[j] == 1:
        distance = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
        if distance < 10:
            penalty += (10 - distance) ** 2
            count += 1

    if count != 0:
        penalty /= count

    return penalty


def evaluate(p):
    score, feasible = calculate_feasibility(p)
    k = p[0] * p[1] / (25 * math.pi)

    if feasible is False:
        score **= 2

    # Number of cookies, area, feasility
    return [45 - np.sum(p[4]), max(100, (p[0] * p[1])), score, feasible]


def mutate(p):
    p[0] = np.clip(p[0] + step_size[0] * np.random.normal(), 10, 100)
    p[1] = np.clip(p[1] + step_size[1] * np.random.normal(), 10, 35)
    p[2] = np.array([np.clip(x + step_size[2] * np.random.normal(), 5, p[0] - 5) for x in p[2]])
    p[3] = np.array([np.clip(x + step_size[3] * np.random.normal(), 5, p[1] - 5) for x in p[3]])
    p[4] = np.array([x if np.random.uniform(0, 1) > step_size[4] else int(not x) for x in p[4]])
    return p


def discrete_recombination(P):
    offsprings = []

    # Repeat p_size times.
    for _ in range(p_size):
        if np.random.uniform() > crossover_rate:
            offsprings.append(P[np.random.randint(0, len(P))])
            continue
        # Sample random individuals
        idxs = np.random.randint(0, len(P), 2)
        individuals = [P[idxs[0]], P[idxs[1]]]

        # Means for a, b
        offspring = []
        offspring.append(np.mean([individuals[0][0], individuals[1][0]]))
        offspring.append(np.mean([individuals[0][1], individuals[1][1]]))

        # Take either a circle from a or from b
        parent_index = np.random.randint(low=0, high=len(idxs), size=len(individuals[0][2]))

        cookies = np.array([[individuals[p_i][2][i], individuals[p_i][3][i], individuals[p_i][4][i]] for i, p_i in enumerate(parent_index)])
        # cookes is of shape (45, 3)
        offspring.append(cookies[:, 0])
        offspring.append(cookies[:, 1])
        offspring.append(cookies[:, 2])

        offsprings.append(offspring)

    return offsprings


def selection(P, scores):
    # Basic u,l tournament selection.
    tournament = np.array([(x, y) for x, y in zip(P, scores)], dtype='object')
    np.random.shuffle(tournament)
    winners = []
    performances = []
    for _ in range(p_size):
        a = tournament[np.random.randint(0, high=len(population))]
        b = tournament[np.random.randint(0, high=len(population))]

        winning = [a, b][np.argmin([a[1], b[1]])]

        winners.append(winning[0].copy())
        performances.append(winning[1])

    return winners, np.array(performances)


def proportional_selection(P, scores):
    scores = np.array(scores)
    scores_padded = np.max(scores) - scores + 0.0001
    probabilities = scores_padded / np.sum(scores_padded)

    idxs = np.random.choice(len(P), size=p_size, p=probabilities)

    return np.array(P, dtype='object')[idxs], scores[idxs]


def calculate_crowding(scores):
    # Crowding is based on a vector for each individual
    # All dimension is normalised between low and high. For any one dimension, all
    # solutions are sorted in order low to high. Crowding for chromsome x
    # for that score is the difference between the next highest and next
    # lowest score. Total crowding value sums all crowding for all scores

    population_size = len(scores[:, 0])
    number_of_scores = len(scores[0, :])

    # create crowding matrix of population (row) and score (column)
    crowding_matrix = np.zeros((population_size, number_of_scores))

    # normalise scores (ptp is max-min)
    ptp = scores.ptp(0)
    if 0 in ptp:
        idxs = np.argwhere(ptp == 0)
        ptp[idxs] = 1

    normed_scores = (scores - scores.min(0)) / ptp

    # calculate crowding distance for each score in turn
    for col in range(number_of_scores):
        crowding = np.zeros(population_size)

        # end points have maximum crowding
        crowding[0] = 1
        crowding[population_size - 1] = 1

        # Sort each score (to calculate crowding between adjacent scores)
        sorted_scores = np.sort(normed_scores[:, col])

        sorted_scores_index = np.argsort(
            normed_scores[:, col])

        # Calculate crowding distance for each individual
        crowding[1:population_size - 1] = (sorted_scores[2:population_size] -
                                           sorted_scores[0:population_size - 2])

        # resort to orginal order (two steps)
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowding[re_sort_order]

        # Record crowding distances
        crowding_matrix[:, col] = sorted_crowding

    # Sum crowding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances


def normalize_scores(evaluation):
    # ptp = evaluation.ptp(0)
    # if 0 in ptp:
    #     idxs = np.argwhere(ptp == 0)
    #     ptp[idxs] = 1
    # evaluation_normed = (evaluation - evaluation.min(0)) / ptp
    # return np.mean(evaluation_normed, axis=1)
    return np.mean(evaluation, axis=1)
    # return calculate_crowding(evaluation)


if __name__ == '__main__':
    seconds = time.time()

    # init population
    t = 0

    population = init_population(p_size)
    evaluation = np.array([evaluate(x) for x in population])
    evaluation = evaluation[:, :-1]
    score = normalize_scores(evaluation)

    tq = trange(t_max)
    for t in tq:
        # Recombine
        # r_population = discrete_recombination(population)

        # Mutate
        m_population = [mutate(p) for p in population]

        # Evaluate
        evaluation = np.array([evaluate(x) for x in m_population])

        feasibility = evaluation[:, -1]
        evaluation = evaluation[:, :-1]

        n_score = normalize_scores(evaluation)

        # Select
        population, score = proportional_selection([*m_population, *population], [*n_score, *score])

        infeasible_solutions = np.argwhere(feasibility == False).flatten()

        tq.set_description(f't = {t}, min score = {round(np.min(score), 3)}, avg score = {round(np.mean(score), 3)}')
        t += 1

    delta = time.time() - seconds
    print(f'done in {delta} seconds.')
    plot_solution(population[np.argmin(score)])

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
