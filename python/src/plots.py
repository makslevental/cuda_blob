from collections import defaultdict
from glob import glob
from operator import itemgetter
from pprint import pprint
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import tikzplotlib
import os


def plot_focus_res(df):
    # df = pd.read_csv("/home/max/dev_projects/cuda_blob/python/tests/image_focus_res.csv")
    # df['focus'] = df['focus']*1000
    print(df)
    x = np.abs(df["focus"] - 0.005399)
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
    b, c = popt

    print(np.corrcoef(x, np.log(y)))

    plt.plot(
        np.sort(x),
        func(np.sort(x), *popt),
        "r-",
        label=f"log(#blobs)$\\approx {b:.2f}x + {c:.2f}$",
    )
    plt.legend()
    # plt.show()
    tikzplotlib.save("blob_count_fit.tex")


def plot_whole():
    vals = defaultdict(list)
    for log_fp in glob("logs/*.log"):
        # GPU whole thing time 1045.791ms
        n_gpus, n_bins, _, _ = map(
            int, os.path.splitext(os.path.split(log_fp)[-1])[0].split("_")
        )
        med_whole_time = np.median(
            [
                float(l.split()[-1].replace("ms", ""))
                for l in open(log_fp).readlines()
                if "whole" in l
            ][1:]
        )
        vals[n_gpus].append((n_bins, med_whole_time))

    fig, ax = plt.subplots()
    for k, v in sorted(vals.items()):
        v.sort()
        # ax.scatter([n_bins for n_bins, _ in v], [avg for _, avg in v], label=f"{k} gpu(s)")
        ax.plot(
            [n_bins for n_bins, _ in v],
            [avg for _, avg in v],
            label=f"{k} gpu(s)",
            marker=".",
            mfc="none",
        )
    ax.legend()
    ax.grid()
    ax.set_ylabel("runtime (ms)")
    ax.set_xlabel("$n$ bins")
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax.set_yscale("log")
    tikzplotlib.save("nbin_vs_gpu.tex", figure=fig)
    # fig.show()


def plot_log_mpi():
    vals = defaultdict(list)
    for log_fp in glob("logs/*.log"):
        # GPU whole thing time 1045.791ms
        n_gpus, n_bins, _, _ = map(
            int, os.path.splitext(os.path.split(log_fp)[-1])[0].split("_")
        )
        med_whole_time = np.median(
            [
                float(l.split()[-1].replace("ms", ""))
                for l in open(log_fp).readlines()
                if "gather" in l
            ][1:]
        )
        vals[n_gpus].append((n_bins, med_whole_time))

    fig, ax = plt.subplots()
    for k, v in sorted(vals.items()):
        v.sort()
        # ax.scatter([n_bins for n_bins, _ in v], [avg for _, avg in v], label=f"{k} gpu(s)")
        ax.plot(
            [n_bins for n_bins, _ in v],
            [avg for _, avg in v],
            label=f"{k} gpu(s)",
            marker=".",
            mfc="none",
        )
    ax.legend()
    ax.grid()
    ax.set_ylabel("runtime (ms)")
    ax.set_xlabel("$n$ bins")
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # ax.set_yscale("log")
    # tikzplotlib.save("gather_vs_nbin.tex", figure=fig)
    fig.show()


def plot_whole_2():
    vals = defaultdict(list)
    for log_fp in glob("logs2/*.log"):
        # GPU whole thing time 1045.791ms
        n_gpus, _, resize, _ = map(
            int, os.path.splitext(os.path.split(log_fp)[-1])[0].split("_")
        )
        med_whole_time = np.median(
            [
                float(l.split()[-1].replace("ms", ""))
                for l in open(log_fp).readlines()
                if "whole" in l
            ][1:]
        )
        vals[n_gpus].append((resize, med_whole_time))

    fig, ax = plt.subplots()
    for k, v in sorted(vals.items()):
        v.sort()
        ax.plot(
            [n_bins for n_bins, _ in v],
            [avg for _, avg in v],
            label=f"{k} gpu(s)",
            marker=".",
            mfc="none",
        )
        # ax.plot([n_bins for n_bins, _ in v], [avg for _, avg in v], label=f"{k} gpu(s)", marker='-o')
    ax.legend()
    ax.grid()
    ax.set_ylabel("runtime (ms)")
    ax.set_xlabel("resolution")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # ax.set_yscale("log")
    tikzplotlib.save("res_vs_gpu.tex", figure=fig)
    # fig.show()


def plot_log_mpi_2():
    vals = defaultdict(list)
    for log_fp in glob("logs2/*.log"):
        # GPU whole thing time 1045.791ms
        n_gpus, _, resize, _ = map(
            int, os.path.splitext(os.path.split(log_fp)[-1])[0].split("_")
        )
        med_whole_time = np.median(
            [
                float(l.split()[-1].replace("ms", ""))
                for l in open(log_fp).readlines()
                if "gather" in l
            ][1:]
        )
        vals[n_gpus].append((resize, med_whole_time))

    fig, ax = plt.subplots()
    for k, v in sorted(vals.items()):
        v.sort()
        ax.plot(
            [n_bins for n_bins, _ in v],
            [avg for _, avg in v],
            label=f"{k} gpu(s)",
            marker=".",
            mfc="none",
        )
        # ax.plot([n_bins for n_bins, _ in v], [avg for _, avg in v], label=f"{k} gpu(s)", marker='-o')
    ax.legend()
    ax.grid()
    ax.set_ylabel("runtime (ms)")
    ax.set_xlabel("resolution (px)")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # ax.set_yscale("log")
    tikzplotlib.save("gather_vs_res.tex", figure=fig)
    # fig.show()


def plot_stacked_bar():
    vals = defaultdict(list)
    for log_fp in glob("logs3/*.log"):
        # GPU whole thing time 1045.791ms
        n_gpus, n_bins, resize, _ = map(
            int, os.path.splitext(os.path.split(log_fp)[-1])[0].split("_")
        )
        if resize != 1024: continue
        lines = open(log_fp).readlines()
        med_filter_time = np.nan_to_num(np.median(
            [
                float(l.split()[-1].replace("ms", ""))
                for l in lines
                if "GPU filtered imgs" in l or "GPU filter imgs time" in l
            ][1:]
        ))
        med_gather_time = np.nan_to_num(np.median(
            [
                float(l.split()[-1].replace("ms", ""))
                for l in lines
                if "gather filtered" in l
            ][1:]
        ))
        med_dog_time = np.nan_to_num(np.median(
            [
                float(l.split()[-1].replace("ms", ""))
                for l in lines
                if "dog time" in l
            ][1:]
        ))
        med_maxima_time = np.nan_to_num(np.median(
            [
                float(l.split()[-1].replace("ms", ""))
                for l in lines
                if "local maxima" in l
            ][1:]
        ))
        vals[n_bins].append((n_gpus,
                             med_filter_time,
                             med_gather_time,
                             med_dog_time,
                             med_maxima_time,
                             ))

    width = 1
    colors = ["red", "green", "blue", "orange", "yellow", "purple", "brown", "grey"]
    fig, ax = plt.subplots()
    for i, (n_bins, v) in enumerate(sorted(vals.items())):
        # if n_bins % 2: continue
        v.sort()
        n_gpus = list(map(itemgetter(0), v))
        med_filter_time = np.array(list(map(itemgetter(1), v)))
        med_gather_time = np.array(list(map(itemgetter(2), v)))
        med_dog_time = np.array(list(map(itemgetter(3), v)))
        med_maxima_time = np.array(list(map(itemgetter(4), v)))
        ax.bar(
            [n_bins+(width+0.02)*i for i in n_gpus],
            med_filter_time,
            width,
            label="Filter" if i == 0 else None,
            color="green",
            edgecolor='black',# hatch=".",
            # color=[colors[i-1] for i in n_gpus],
            # tick_label=list(map(str, n_gpus))
        )
        ax.bar(
            [n_bins+(width+0.02)*i for i in n_gpus],
            med_gather_time,
            width,
            bottom=med_filter_time,
            label="Gather" if i == 0 else None,
            color="red",
            edgecolor='black',# hatch="//",
            # color=[colors[i-1] for i in n_gpus],
            # tick_label=list(map(str, n_gpus))
        )
        ax.bar(
            [n_bins+(width+0.02)*i for i in n_gpus],
            med_dog_time,
            width,
            bottom=med_gather_time+med_filter_time,
            label="DoG" if i == 0 else None,
            color="blue",
            edgecolor='black',# hatch="*",
            # color=[colors[i-1] for i in n_gpus],
            # tick_label=list(map(str, n_gpus))
        )
        ax.bar(
            [n_bins+(width+0.02)*i for i in n_gpus],
            med_maxima_time,
            width,
            bottom=med_dog_time+med_gather_time+med_filter_time,
            label="Maxima" if i == 0 else None,
            color="yellow",
            edgecolor='black',# hatch=".",
            # color=[colors[i-1] for i in n_gpus],
            # tick_label=list(map(str, n_gpus))
        )

    ax.legend()
    ax.grid()
    # ax.set_xticks([n_bins for n_bins, v in sorted(vals.items()) if n_bins % 4 == 0])
    # ax.set_xticklabels(labels)
    ax.set_ylabel("runtime (ms)")
    ax.set_xlabel("$n$ bins")
    # ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # ax.set_yscale("log")
    tikzplotlib.save("stacked_runtime.tex", figure=fig, standalone=False)
    # fig.show()


# def plot_log3():
#     vals = defaultdict(list)
#     for log_fp in glob("logs2/*.log"):
#         # GPU whole thing time 1045.791ms
#         n_gpus, _, resize, _ = map(
#             int, os.path.splitext(os.path.split(log_fp)[-1])[0].split("_")
#         )
#         avg_whole_time = np.mean(
#             [
#                 float(l.split()[-1].replace("ms", ""))
#                 for l in open(log_fp).readlines()
#                 if "whole" in l
#             ][1:]
#         )
#         vals[n_gpus].append((resize, avg_whole_time))
#
#     fig, ax = plt.subplots()
#     for k, v in sorted(vals.items()):
#         v.sort()
#         ax.plot([n_bins for n_bins, _ in v[:-1]], [avg for _, avg in v[:-1]], label=f"{k} gpu(s)", marker='o', mfc='none')
#         # ax.plot([n_bins for n_bins, _ in v], [avg for _, avg in v], label=f"{k} gpu(s)", marker='-o')
#     ax.legend()
#     ax.grid()
#     ax.set_ylabel("ms")
#     ax.set_xlabel("res")
#     ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#     ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#     # ax.set_yscale("log")
#     # tikzplotlib.save("res_vs_gpu_2.tex", figure=fig)
#     fig.show()


if __name__ == "__main__":
    plot_stacked_bar()
    plot_whole()
    plot_whole_2()
    # plot_log3()
    # plot_log_mpi()
    # plot_log_mpi_2()

# rigor
# science
# ml
# scale
# distributed hpc
