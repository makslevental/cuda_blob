import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def plot_focus_res():
    df = pd.read_csv("/home/max/dev_projects/cuda_blob/python/tests/image_focus_res.csv")
    x = np.abs(df["focus"] - 5.399)
    y = df["blobs"]
    plt.scatter(x, y)
    plt.xlabel("|focus_depth - 5.399|")
    plt.ylabel("# blobs")
    plt.title('blob count vs "out of focus"')
    # plt.show()
    # y ≈ a exp(-b*x) + c
    # log(y-c) = log(a) + -b*x
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, pcov = curve_fit(func, x, y)
    a,b,c = popt

    plt.plot(np.sort(x), func(np.sort(x), *popt), 'r-', label=f"#blobs $\\approx {a:.2f}e^{{-{b:.2f}x}} {c:.2f}$")
    # plt.yscale("log")
    plt.legend()
    plt.show()

    plt.scatter(x, np.log(y))
    plt.xlabel("|focus_depth - 5.399|")
    plt.ylabel("log(# blobs)")
    plt.title('blob count vs "out of focus"')
    # plt.show()
    def func(x, b, c):
        return b * x + c

    popt, pcov = curve_fit(func, x, np.log(y))
    b,c = popt

    print(np.corrcoef(x, np.log(y)))

    plt.plot(np.sort(x), func(np.sort(x), *popt), 'r-', label=f"log(#blobs)$\\approx {b:.2f}x + {c:.2f}$")
    plt.legend()
    plt.show()