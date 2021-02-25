import collections
import math
from bisect import bisect_left
from functools import lru_cache
from typing import Tuple

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import xmltodict
from PIL import Image, ImageChops
from scipy import spatial
from skimage.feature.blob import _blob_overlap

from src.kernels import resize_images_interpolate_bilinear
from src.profiling import GPUTimer


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


def read_tiff_metadata(fp):
    img = tifffile.TiffFile(fp)
    fibicsXML = img.pages.pages[0].tags.get(51023)
    assert fibicsXML.name == "FibicsXML"
    xml = fibicsXML.value
    xml_dict = xmltodict.parse(xml)["Fibics"]
    left, right, top, bottom = map(
        int,
        [
            xml_dict["Image"]["BoundingBox.Left"],
            xml_dict["Image"]["BoundingBox.Right"],
            xml_dict["Image"]["BoundingBox.Top"],
            xml_dict["Image"]["BoundingBox.Bottom"],
        ],
    )
    return float(xml_dict["Scan"]["Focus"]), (left, right, top, bottom)


def trim(img) -> Image:
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, scale=2.0, offset=-100)
    bbox = diff.getbbox()
    print(bbox)
    if bbox:
        return img.crop(bbox)
    else:
        return img


PINNED_MEMORY_POOL = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(PINNED_MEMORY_POOL.malloc)


def pin_memory(array):
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret


PINNED_ARR = None


# (426, 474, 4646, 5239)
def load_img_to_gpu(fp, resize=(1024, 1024)) -> cp.ndarray:
    global PINNED_ARR

    # TODO(max): pinned memory?
    with GPUTimer("open"):
        # img = Image.open(fp)
        img = tifffile.imread(fp)


    with GPUTimer("crop"):
        # img = trim(img)
        _, (left, right, upper, lower) = read_tiff_metadata(fp)
        img = img[upper : lower + 1, left : right + 1]

    with GPUTimer("pin memory"):
        if PINNED_ARR is None or PINNED_ARR.shape != img.shape:
            PINNED_ARR = pin_memory(img)
        PINNED_ARR[...] = img
        img = PINNED_ARR

    with GPUTimer("copy img to gpu"):
        img = cp.array(img) / cp.float32(255.0)

    if resize is not None:
        with GPUTimer("resize"):
            img = gpu_resize(img, resize)

    return img


# https://github.com/chainer/chainer/blob/v7.7.0/chainer/functions/array/resize_images.py#L174
@lru_cache(maxsize=None)
def compute_indices_and_weights(out_size, in_size, mode, align_corners):
    out_H, out_W = out_size
    H, W = in_size
    if mode == "bilinear":
        if align_corners:
            v = cp.linspace(0, H - 1, num=out_H, dtype=cp.float)
            u = cp.linspace(0, W - 1, num=out_W, dtype=cp.float)
        else:
            y_scale = H / out_H
            x_scale = W / out_W
            v = (cp.arange(out_H, dtype=cp.float) + 0.5) * y_scale - 0.5
            v = cp.maximum(v, 0)
            u = (cp.arange(out_W, dtype=cp.float) + 0.5) * x_scale - 0.5
            u = cp.maximum(u, 0)
        vw, v = cp.modf(v)
        uw, u = cp.modf(u)
    elif mode == "nearest":
        y_scale = H / out_H
        x_scale = W / out_W
        v = cp.minimum(cp.floor(cp.arange(out_H, dtype=cp.float) * y_scale), H - 1)
        u = cp.minimum(cp.floor(cp.arange(out_W, dtype=cp.float) * x_scale), W - 1)
        vw = cp.zeros_like(v)
        uw = cp.zeros_like(u)
    return v, u, vw, uw


# https://github.com/chainer/chainer/blob/v7.7.0/chainer/functions/array/resize_images.py#L82
def interpolate_bilinear_gpu(x, v, u, vw, uw):
    H, W = x.shape
    out_H, out_W = v.shape
    y = cp.empty((out_H, out_W), dtype=x.dtype)

    resize_images_interpolate_bilinear(x, v, u, vw, uw, H, W, out_H * out_W, y)
    return y


# https://github.com/chainer/chainer/blob/v7.7.0/chainer/functions/array/resize_images.py#L224
def gpu_resize(
    img: cp.ndarray, resize=(1024, 1024), mode="bilinear", align_corners=True
):
    v, u, vw, uw = compute_indices_and_weights(resize, img.shape, mode, align_corners)
    v = v.astype(cp.intp)
    u = u.astype(cp.intp)
    vw = vw.astype(img.dtype)
    uw = uw.astype(img.dtype)

    v, u, vw, uw = cp.broadcast_arrays(v[:, None], u[None, :], vw[:, None], uw[None, :])

    y = interpolate_bilinear_gpu(img, v, u, vw, uw)
    return y


dim2 = collections.namedtuple("dim2", "x y")
dim3 = collections.namedtuple("dim3", "x y z")


@lru_cache(maxsize=None)
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


def stretch_composite_histogram(image: cp.ndarray, saturation_pct=0.35):
    bin_density, bin_edges = cp.histogram(image, bins=256, density=True)
    bin_density /= bin_density.sum()
    bin_cdf = cp.cumsum(bin_density)
    h_min_idx = bisect_left(bin_cdf, saturation_pct // 100)
    h_max_idx = bisect_left(bin_cdf, 1 - saturation_pct / 100)

    imin, imax = bin_edges[h_min_idx], bin_edges[h_max_idx]
    omin, omax = map(cp.float32, (0.0, 1.0))
    image = cp.clip(image, imin, imax)
    image = (image - imin) / (imax - imin)

    return image * (omax - omin) + omin


@lru_cache(maxsize=None)
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
