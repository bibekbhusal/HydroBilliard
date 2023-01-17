import numba
import numpy as np
from numba import jit, prange
from numba import types as t


def cal_boundary_length(length: int):
    return [[-length / 2.0, length / 2.0], [-length / 2.0, length / 2.0]]


def find_out_of_bounds(x_co_ordinates, y_co_ordinates, border_path):
    """
    Finds and return particles, pairs of (x,y) from (x_x_co_ordinates, y_co_ordinates) that crossed the given border
    path.
    """
    in_bound, out_bound = get_inbound_outbound(x_co_ordinates, y_co_ordinates, border_path)
    return out_bound


def get_inbound_outbound(x_co_ordinates, y_co_ordinates, border_path):
    bound_status = _get_bound_status(x_co_ordinates, y_co_ordinates, border_path)
    in_bound = np.where(bound_status == True)[0]
    out_bound = np.where(bound_status == False)[0]
    return in_bound, out_bound


def _get_bound_status(x_co_ordinates, y_co_ordinates, border_path):
    return border_path.contains_points(np.vstack((x_co_ordinates, y_co_ordinates)).T)


@jit(t.UniTuple(t.float64, 2)(t.float64, t.float64))
def rand_cos_norm(norm_x, norm_y):
    """
    Returns a vector selected from a cosine distribution relative to an input normal vector.
    """
    # An angle between -pi/2 and pi/2 following cosine distribution
    theta = np.pi * (np.random.rand() - 0.5)
    # TODO: ????
    while np.random.rand() > np.cos(theta):
        theta = np.pi * (np.random.rand() - 0.5)
    v_x_new = -norm_x * np.cos(theta) - norm_y * np.sin(theta)
    v_y_new = -norm_y * np.cos(theta) + norm_x * np.sin(theta)
    return v_x_new, v_y_new


# Expectation is len(x_co_ordinates) == len(y_co_ordinates)
@jit(t.UniTuple(t.float64[:], 3)(t.float64[:], t.float64[:]), nopython=True, parallel=True)
def poly_norms(x_co_ordinates, y_co_ordinates):
    dx = x_co_ordinates[1:] - x_co_ordinates[:-1]
    dy = y_co_ordinates[1:] - y_co_ordinates[:-1]

    lengths = np.sqrt(dx ** 2 + dy ** 2)
    norm_x = -dy / lengths
    norm_y = dx / lengths

    return norm_x, norm_y, lengths


@jit(t.UniTuple(t.float64, 2)(t.float64, t.float64, t.float64, t.float64))
def mirror_norm(norm_x, norm_y, v_x_in, v_y_in):
    vec_proj = norm_x * v_x_in + norm_y * v_y_in
    v_x_out = v_x_in - 2 * norm_x * vec_proj
    v_y_out = v_y_in - 2 * norm_y * vec_proj

    return v_x_out, v_y_out


@jit(t.boolean(t.float64, t.float64, t.float64, t.float64, t.float64, t.float64, t.float64, t.float64), nopython=True)
def seg_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Checks if two points [x3, y3] and [x4, y4] intersects a line/segment with endpoints [x1, y1] and [x2, y2].
    """
    x13 = x1 - x3
    y13 = y1 - y3
    x21 = x2 - x1
    y21 = y2 - y1
    x43 = x4 - x3
    y43 = y4 - y3

    ts = (x13 * y21 - y13 * x21) / (x43 * y21 - y43 * x21)
    us = (x13 * y43 - y13 * x43) / (x43 * y21 - y43 * x21)

    return (ts >= 0) and (ts <= 1) and (us >= 0) and (us <= 1)


@jit(t.int64[:](t.float64[:], t.float64[:], t.float64[:], t.float64[:]), nopython=True)
def seg_cross_poly(poly_x, poly_y, p1, p2):
    """
    Calculates and return edges of the given polygon cross when a particle movies from p1 to p2.
    """
    crossed = []
    total_edges = len(poly_x) - 1

    for edge_index in range(total_edges):
        if seg_intersect(poly_x[edge_index], poly_y[edge_index], poly_x[edge_index + 1],
                         poly_y[edge_index + 1], p1[0], p1[1], p2[0], p2[1]):
            crossed.append(edge_index)
    return np.array(crossed)
