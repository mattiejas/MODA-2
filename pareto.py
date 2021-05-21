import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def evaluate(p: np.array) -> np.array:
    return np.random.uniform(0, 45, size=len(p))


def identify_pareto(p: np.array) -> np.array:
    population_ids = np.arange(len(p))
    pareto_front = np.ones(len(p), dtype=bool)

    for i in range(len(p)):
        for j in range(len(p)):

            if all(p[i] >= p[j]) and any(p[i] > p[j]):
                pareto_front[i] = 0
                break

    return population_ids[pareto_front]


def plot_pereto(population):
    pareto = identify_pareto(population)
    pareto_front = population[pareto]

    pareto_front_df = pd.DataFrame(pareto_front)
    pareto_front_df.sort_values(0, inplace=True)
    pareto_front = pareto_front_df.values

    x_all = population[:, 0]
    y_all = population[:, 1]
    x_pareto = pareto_front[:, 0]
    y_pareto = pareto_front[:, 1]

    plt.scatter(x_all, y_all)
    plt.plot(x_pareto, y_pareto, color='r', label='pareto front')
    plt.legend()
    plt.title("Pareto front for cookie cutting problem")
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()


if __name__ == '__main__':
    plot_pereto(np.random.random((20, 2)))
