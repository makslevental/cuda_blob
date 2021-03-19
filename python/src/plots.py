import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import tikzplotlib

def plot_focus_res(df):
    # df = pd.read_csv("/home/max/dev_projects/cuda_blob/python/tests/image_focus_res.csv")
    # df['focus'] = df['focus']*1000
    print(df)
    x = np.abs(df["focus"] - .005399)
    y = df["blobs"]


    # plt.scatter(x, y)
    # plt.xlabel("|focus_depth - .005399|")
    # plt.ylabel("# blobs")
    # plt.title('blob count vs "out of focus"')
    # # plt.show()
    # # y ≈ a exp(-b*x) + c
    # # log(y-c) = log(a) + -b*x
    # def func(x, a, b):
    #     return a * np.exp(-b * x)
    #
    # popt, pcov = curve_fit(func, x, y)
    # a,b = popt
    #
    # plt.plot(np.sort(x), func(np.sort(x), *popt), 'r-', label=f"#blobs $\\approx {a:.2f}e^{{-{b:.2f}x}}$")
    # # plt.yscale("log")
    # plt.legend()
    # # plt.show()

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
    # plt.show()
    tikzplotlib.save("blob_count_fit.tex")
