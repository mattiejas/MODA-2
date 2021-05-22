import matplotlib.pyplot as plt

def plot_solution(p):
    a, b, x, y, s = p
    fig = plt.figure()

    for i in range(len(s)):
        if s[i]:
            circle1 = plt.Circle((x[i], y[i]), 5, color='k', linewidth=1, fill=False)
            plt.gca().add_patch(circle1)

    plt.xlim((0, a))
    plt.ylim((0, b))

    ax = fig.axes[0]
    ax.set_aspect('equal', 'box')

    plt.show()