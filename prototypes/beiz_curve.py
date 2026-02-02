import numpy as np
import matplotlib.pyplot as plt
import math

def bezier_curve(points, n=100):
    t = np.linspace(0, 1, n)
    curve = np.zeros((n, 2))
    degree = len(points) - 1

    for i in range(degree + 1):
        binomial = math.comb(degree, i)
        curve += binomial * ((1 - t) ** (degree - i))[:, None] * (t ** i)[:, None] * points[i]

    return curve

if __name__ == "__main__":

    control_points = np.array([
        [0, 0],
        [1, 2],
        [3, 3],
        [4, 0]
    ])

    curve = bezier_curve(control_points)

    cp2 = np.array([ [0, 0], [6,2], [4, 0]])
    c2 = bezier_curve(cp2)

    plt.plot(curve[:, 0], curve[:, 1], label='Bézier Curve')
    plt.plot(c2[:, 0], c2[:, 1])
    plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Control Points')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.title('Cubic Bézier Curve')
    plt.show()
