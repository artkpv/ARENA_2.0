# %%
import os
import sys
import torch as t
from torch import Tensor
import einops
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"

# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    res = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    res[:, 1, 0] = 1.0
    t.linspace(-y_limit, y_limit, num_pixels, out=res[:, 1, 1])
    return res

# %%

if MAIN:
    rays1d = make_rays_1d(9, 10.0) * 3
    segments = t.tensor([
        [[1.0, -12.0, 0.0], [1, -6.0, 0.0]],   # Yes. At two points.
        [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]],  # Doesn't
        [[2, 12.0, 0.0], [2, 21.0, 0.0]]    # Yes.
    ])
    fig = render_lines_with_plotly(rays1d, segments)

# %%

def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    if set(ray[:, 2].tolist()) != set(segment[:, 2].tolist()):
        # TODO: solve only for 3d?
        return False
    A = t.empty(2, 2)
    A[0, :] = ray[1, 0:2]
    A[1, :] = segment[0, 0:2] - segment[1, 0:2]
    A = A.T
    B = (segment[0:1, 0:2] - ray[0:1, 0:2]).T
    if t.linalg.det(A) != 0:
        u, v = t.linalg.solve(A, B)
        return u >= 0 and 0 <= v <= 1
    return False


if MAIN:
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %%
