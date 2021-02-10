import collections
import math
from bisect import bisect_left
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
from PIL.Image import BILINEAR
from scipy import spatial
from skimage import exposure
from skimage.feature.blob import _blob_overlap


def nd_to_text(A, w=None):
    if A.ndim == 1:
        if w is None:
            return str(A)
        else:
            s = "["
            for i, AA in enumerate(A[:-1]):
                s += str(AA) + " " * (max(w[i], len(str(AA))) - len(str(AA)) + 1)
            s += (
                str(A[-1])
                + " " * (max(w[-1], len(str(A[-1]))) - len(str(A[-1])))
                + "] "
            )
    elif A.ndim == 2:
        w1 = [max([len(str(s)) for s in A[:, i]]) for i in range(A.shape[1])]
        w0 = sum(w1) + len(w1) + 1
        s = "\u250c" + "\u2500" * w0 + "\u2510" + "\n"
        for AA in A:
            s += " " + nd_to_text(AA, w=w1) + "\n"
        s += "\u2514" + "\u2500" * w0 + "\u2518"
    elif A.ndim == 3:
        h = A.shape[1]
        s1 = "\u250c" + "\n" + ("\u2502" + "\n") * h + "\u2514" + "\n"
        s2 = "\u2510" + "\n" + ("\u2502" + "\n") * h + "\u2518" + "\n"
        strings = [nd_to_text(a) + "\n" for a in A]
        strings.append(s2)
        strings.insert(0, s1)
        s = "\n".join("".join(pair) for pair in zip(*map(str.splitlines, strings)))
    return s


def print_nd_array(arr: np.ndarray, round=3):
    print(nd_to_text(arr.round(round)))


def trim(img) -> Image:
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, scale=2.0, offset=-100)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)
    else:
        return img


def load_img(fp, resize=None) -> np.ndarray:
    # TODO(max): pinned memory?
    # convert to I so that there's no clipping (float for CUDA)
    img = trim(Image.open(fp)).convert("I").convert("F")
    if resize is not None:
        img = img.resize(resize, resample=BILINEAR)

    return np.array(img) / 255.0


dim2 = collections.namedtuple("dim2", "x y")
dim3 = collections.namedtuple("dim3", "x y z")


def get_grid_block_dims(batch_size, img_h, img_w, num_threads=32) -> Tuple[dim3, dim2]:
    n_blocks_h, n_blocks_w = img_h // num_threads, img_w // num_threads
    if n_blocks_w % batch_size or n_blocks_w == 0:
        n_blocks_w += 1
    if n_blocks_h % batch_size or n_blocks_h == 0:
        n_blocks_h += 1

    return (n_blocks_w, n_blocks_h, batch_size), (num_threads, num_threads)


def prune_blobs(
    *,
    blobs_array: np.ndarray,
    overlap: float,
    local_maxima: np.ndarray = None,
    sigma_dim: int = 1,
) -> np.ndarray:
    """Find non-overlapping blobs
    Parameters
    ----------
    blobs_array: n x 3 where first two cols are x,y coords and third col is blob radius
    overlap: minimum area overlap in order to prune one of the blobs
    local_maxima: optional maxima values at peaks. if included then stronger maxima will be chosen on overlap
    sigma_dim: which column in blobs_array has the radius
    Returns
    -------
    blobs_array: non-overlapping blobs
    """

    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - sigma_dim)
    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array

    for (i, j) in pairs:
        blob1, blob2 = blobs_array[i], blobs_array[j]
        blob_overlap = _blob_overlap(blob1, blob2, sigma_dim=sigma_dim)
        if blob_overlap > overlap:
            # if local maxima then pick stronger maximum
            if local_maxima is not None:
                if local_maxima[i] > local_maxima[j]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0
            # else take average
            else:
                blob2[-1] = (blob1[-1] + blob2[-1]) / 2
                blob1[-1] = 0

    return blobs_array[blobs_array[:, -1] > 0]


def make_fig(image, blobs=None, title=None, dpi=96):
    px, py = image.shape
    fig = plt.figure(figsize=(py / np.float(dpi), px / np.float(dpi)))
    if title is None:
        dims = [0.0, 0.0, 1.0, 1.0]
    else:
        dims = [0.0, 0.0, 1.0, 0.95]
    ax = fig.add_axes(dims, yticks=[], xticks=[], frame_on=False)
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontsize=20)
    if blobs is not None:
        for y, x, r in blobs:
            c = plt.Circle((x, y), r, color="red", linewidth=1, fill=False)
            ax.add_patch(c)
    return fig


def make_fig_square(image, blobs=None, title=None, dpi=96):
    px, py = image.shape
    fig = plt.figure(figsize=(py / np.float(dpi), px / np.float(dpi)))
    if title is None:
        dims = [0.0, 0.0, 1.0, 1.0]
    else:
        dims = [0.0, 0.0, 1.0, 0.95]
    ax = fig.add_axes(dims, yticks=[], xticks=[], frame_on=False)
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontsize=20)
    if blobs is not None:
        for y, x, r in blobs:
            c = plt.Rectangle((x, y), r, r, color="red", linewidth=1, fill=False)
            ax.add_patch(c)
    return fig


def make_hist(vals, title=None, use_log_scale=False, n_bins=256):
    fig, ax = plt.subplots(tight_layout=True)
    ax.hist(vals, bins=n_bins)
    ax.set_title(title)
    if use_log_scale:
        ax.set_yscale("log")
    return fig


def stretch_composite_histogram(image: np.ndarray, saturation_pct=0.35):
    bin_density, bin_edges = np.histogram(image, bins=256, density=True)
    bin_density /= bin_density.sum()
    bin_cdf = np.cumsum(bin_density)
    h_min_idx = bisect_left(bin_cdf, saturation_pct // 100)
    h_max_idx = bisect_left(bin_cdf, 1 - saturation_pct / 100)
    return exposure.rescale_intensity(
        image, in_range=(bin_edges[h_min_idx], bin_edges[h_max_idx]), out_range="image"
    )


def get_sigmas(
    img_h, img_w, min_sigma, max_sigma, n_sigma_bins, truncate
) -> np.ndarray:
    # calculate sigmas (corresponding to radii)
    sigmas = np.linspace(
        min_sigma,
        max_sigma
        + (max_sigma - min_sigma)
        / (n_sigma_bins - 1),  # go one increment higher so that we include max_sigma
        n_sigma_bins + 1,
    )

    max_radius = int(truncate * max(sigmas) + 0.5)
    assert max_radius < img_h // 2 and max_radius < img_w // 2
    return sigmas


def display_slices(stack):
    for s in stack:
        make_fig(s.get()).show()