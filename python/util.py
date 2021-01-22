from typing import Tuple

import numpy as np
from PIL import Image, ImageChops
from PIL.Image import BILINEAR
import collections


def nd_to_text(A, w=None, h=None):
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
        s = u"\u250c" + u"\u2500" * w0 + u"\u2510" + "\n"
        for AA in A:
            s += " " + nd_to_text(AA, w=w1) + "\n"
        s += u"\u2514" + u"\u2500" * w0 + u"\u2518"
    elif A.ndim == 3:
        h = A.shape[1]
        s1 = u"\u250c" + "\n" + (u"\u2502" + "\n") * h + u"\u2514" + "\n"
        s2 = u"\u2510" + "\n" + (u"\u2502" + "\n") * h + u"\u2518" + "\n"
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


def load_tiff(fp, resize=None) -> np.ndarray:
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

    return dim3(n_blocks_w, n_blocks_h, batch_size), dim2(num_threads, num_threads)
