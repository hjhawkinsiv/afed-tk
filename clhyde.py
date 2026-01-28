import numpy as np

from collections import defaultdict
from datetime import datetime
from numba import njit, types as numba_types

from djset import *
from contour import TemporalContour


@njit
def clhyde(cielab_video, foreground_pixels, delta, labels, label_set, neighbor_offsets):
    depth = cielab_video.shape[0]
    height = cielab_video.shape[1]
    width = cielab_video.shape[2]
    pixels_per_frame = height * width
    pixel_count = depth * pixels_per_frame
    next_label = 1
    label_set_ranks = empty_disjoint_set(numba_types.int64)
    flattened_cielab_video = cielab_video.reshape((pixel_count, 3))

    for f, r, c in foreground_pixels:
        candidate_neighbors = neighbor_offsets + np.array([f, r, c])
        candidate_frames = candidate_neighbors[..., 0]
        candidate_rows = candidate_neighbors[..., 1]
        candidate_columns = candidate_neighbors[..., 2]

        neighbor_indices = np.argwhere(
            (candidate_frames >= 0) &
            (candidate_rows >= 0) & (candidate_rows < height) &
            (candidate_columns >= 0) & (candidate_columns < width)
        ).flatten()

        candidate_neighbors = candidate_neighbors.take(neighbor_indices, axis=0)
        indices = candidate_neighbors[..., 0] * pixels_per_frame + candidate_neighbors[..., 1] * width + \
                  candidate_neighbors[..., 2]
        candidate_labels = labels.take(indices)
        candidate_label_indices = np.argwhere(candidate_labels).flatten()
        candidates = candidate_neighbors.take(candidate_label_indices, axis=0)

        if len(candidates) > 0:
            candidate_indices = candidates[..., 0] * pixels_per_frame + candidates[..., 1] * width + candidates[..., 0]
            candidate_colors = flattened_cielab_video.take(candidate_indices, axis=0)

            differences = candidate_colors - cielab_video[f][r][c]
            differences = np.abs(differences[..., 0]) + np.sqrt(differences[..., 1] ** 2 + differences[..., 2] ** 2)
            min_difference_index = np.argmin(differences)

            if differences[min_difference_index] < delta:
                candidate = candidates[min_difference_index]
                contour_label = labels[candidate[0]][candidate[1]][candidate[2]]
                labels[f][r][c] = contour_label
                siblings = (differences < delta).nonzero()[0]

                for i in siblings:
                    sibling = candidates[i]

                    disjoint_set_union(
                        contour_label,
                        labels[sibling[0]][sibling[1]][sibling[2]],
                        label_set,
                        label_set_ranks
                    )
            else:
                contour_label = next_label
                disjoint_set_add(next_label, label_set)
                next_label += 1
                labels[f][r][c] = contour_label
        else:
            contour_label = next_label
            disjoint_set_add(next_label, label_set)
            next_label += 1
            labels[f][r][c] = contour_label


def cluster(cielab_video: np.ndarray, mask: np.ndarray, neighbor_offsets, threshold: float = 10.0):
    labels = np.zeros_like(mask, dtype=np.int64)
    label_set = empty_disjoint_set(numba_types.int64)
    foreground_pixels = np.argwhere(mask)

    print(f"[{datetime.now()}]::labeling {len(foreground_pixels)} pixels...")

    clhyde(cielab_video, foreground_pixels, threshold, labels, label_set, neighbor_offsets)

    print(f"[{datetime.now()}]::end labeling pixels...")

    components = defaultdict(list)

    for f, r, c in foreground_pixels:
        label = disjoint_set_find(labels[f, r, c], label_set)
        components[label].append((f, r, c))

    print(f"[{datetime.now()}]::generated_contours...")

    for contour in components.values():
        yield TemporalContour(np.array(contour))
