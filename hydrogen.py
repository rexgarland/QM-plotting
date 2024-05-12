import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.special as ss
import numpy as np
import math, sympy


def Laguerre(p, q):
    x = sympy.Symbol("x")
    exp = (-1) ** p * sympy.diff(
        sympy.simplify(sympy.E**x * sympy.diff(sympy.E ** (-x) * x**q, x, q)), x, p
    )
    return lambda x: eval(str(exp))


def SphericalHarmonic(l, m, theta, phi, epsilon):
    # this function uses the associated legendre polynomial evaluator from scipy.special
    return (
        epsilon
        * np.sqrt(
            (2.0 * l + 1)
            * math.factorial(l - np.abs(m))
            / (4 * np.pi * math.factorial(l + np.abs(m)))
        )
        * np.real(np.exp(1.0j * m * phi) * ss.lpmv(m, l, np.cos(theta)))
    )


def Radial(n, l, r, a):
    p = r / (n * a)
    lag = Laguerre(2 * l + 1, n + l)
    # this function uses the associated laguerre polynomial evaluator from scipy.special
    return (
        np.sqrt(
            (2.0 / n / a) ** 3
            * math.factorial(n - l - 1)
            / (2.0 * n * (math.factorial(n + 1)) ** 3)
        )
        * np.exp(-p)
        * (2 * p) ** l
        * lag(2 * p)
    )


def graph(n, l, m, a):

    # parameters for the plot (tuned such that the bohr radius is on the same scale as
    # pixels #)
    resolution = 300
    frame_apothem = 300

    # constant for the spherical harmonics (epsilon)
    if m >= 0:
        eps = (-1) ** m
    else:
        eps = 1

    # create array of data points
    x = np.linspace(-frame_apothem * 1.6, frame_apothem * 1.6, int(resolution * 1.6))
    y = np.linspace(-frame_apothem, frame_apothem, resolution)
    X, Y = np.meshgrid(x, x)
    # create an array of wavefunction values (1e-10 added so that arctan never sees X/0)
    Z = (
        np.abs(
            Radial(n, l, np.sqrt((X**2 + Y**2)), a)
            * SphericalHarmonic(l, m, np.arctan(X / (Y + 1e-10)), 0, eps)
        )
        ** 2
    )
    Z = Z.astype(np.float64)
    Z = np.sqrt(
        Z
    )  # this is done to "raise" the lower, less perceptible values to sight
    # plot the wavefunction in grayscale
    plt.imshow(Z, cmap=cm.Greys_r)
    plt.show()


def main():
    # ask for appropriate quantum numbers
    n = int(input("n? "))
    while n < 1:
        print("Invalid.")
        n = int(input("n? "))
    l = int(input("l? "))
    while l >= n or l < 0:
        print("Invalid.")
        l = int(input("l? "))
    m = int(input("m? "))
    while np.abs(m) > l:
        print("Invalid.")
        m = int(input("m? "))

    # ask for "Bohr Radius," which is tuned in graphWave to display a Bohr radius of
    # approximately "a" pixels
    a = float(input("Bohr Radius (in pixels)? "))
    while a < 0:
        print("Invalid.")
        a = float(input("Bohr Radius (in pixels)? "))

    # graph wavefunction
    graph(n, l, m, a)


if __name__ == "__main__":
    main()
