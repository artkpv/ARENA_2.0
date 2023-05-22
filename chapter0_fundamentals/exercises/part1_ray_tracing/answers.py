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

# import plotly.io as pio
# pio.renderers.default = "vscode"

# %%

if MAIN:
    pass
    #os.chdir(exercises_dir)

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

def intersect_ray_1d(ray: Float[t.Tensor, '... 2 3'], segment: Float[t.Tensor, '... 2 3']) -> Bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    if set(ray[:, 2].tolist()) != set(segment[:, 2].tolist()):
        # TODO: solve only for 3d?
        return False
    A = t.empty(2, 2, dtype=t.float)
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
def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
 # SOLUTION
    NR = rays.size(0)
    NS = segments.size(0)

    # Get just the x and y coordinates
    rays = rays[..., :2]
    segments = segments[..., :2]

    # Repeat rays and segments so that we can compuate the intersection of every (ray, segment) pair
    rays = einops.repeat(rays, "nrays p d -> nrays nsegments p d", nsegments=NS)
    segments = einops.repeat(segments, "nsegments p d -> nrays nsegments p d", nrays=NR)

    # Each element of `rays` is [[Ox, Oy], [Dx, Dy]]
    O = rays[:, :, 0]
    D = rays[:, :, 1]
    assert O.shape == (NR, NS, 2)

    # Each element of `segments` is [[L1x, L1y], [L2x, L2y]]
    L_1 = segments[:, :, 0]
    L_2 = segments[:, :, 1]
    assert L_1.shape == (NR, NS, 2)

    # Define matrix on left hand side of equation
    mat = t.stack([D, L_1 - L_2], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    assert is_singular.shape == (NR, NS)
    mat[is_singular] = t.eye(2)

    # Define vector on the right hand side of equation
    vec = L_1 - O

    # Solve equation, get results
    sol = t.linalg.solve(mat, vec)
    u = sol[..., 0]
    v = sol[..., 1]

    # Return boolean of (matrix is nonsingular, and solution is in correct range implying intersection)
    return ((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=-1)

    # RN = rays.shape[0]
    # SN = segments.shape[0]
    # rays = einops.repeat(rays[...,:2], 'RN p d -> RN SN p d', RN=RN, SN=SN)
    # segments = einops.repeat(segments[...,:2], 'SN p d -> RN SN p d', RN=RN, SN=SN)
    # Os = rays[:,:, 0]
    # Ds = rays[:,:, 1]
    # L_1s = segments[:,:, 0]
    # L_2s = segments[:,:, 1]

    # A = einops.rearrange([Ds, (L_1s - L_2s)], 'm RN SN d -> RN SN m d')
    # #A = t.stack([Ds, (L_1s - L_2s)], dim=-1)
    # EPS=1e-8
    # is_zero = t.linalg.det(A).abs() < EPS
    # A[is_zero] = t.eye(2)
    # B = L_1s - Os
    # s = t.linalg.solve(A, B)
    # u = s[..., 0]
    # v = s[..., 1]
    # return ((u >= 0 & ((0 <= v) & (v <= 1))) & ~is_zero).any(dim=-1)


if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d) 
# %%

# ChatGPT generated:

import unittest

class IntersectRaysTest(unittest.TestCase):

    def test_empty_rays(self):
        rays = t.empty(0, 3, 2)
        segments = t.empty(2, 3, 2)
        result = intersect_rays_1d(rays, segments)
        self.assertTrue(t.all(result == False))

    def test_empty_segments(self):
        rays = t.empty(2, 3, 2)
        segments = t.empty(0, 3, 2)
        result = intersect_rays_1d(rays, segments)
        self.assertTrue(t.all(result == False))

    def test_exact_intersection(self):
        rays = t.tensor([
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 1.0], [2.0, 1.0]],
        ])
        segments = t.tensor([
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 1.0], [2.0, 1.0]],
        ])
        result = intersect_rays_1d(rays, segments)
        self.assertTrue(t.all(result == True))

    def test_singular_segment_not_considered(self):
        rays = t.tensor([
            [[0.0, 0.0], [1.0, 0.0]],
            
        ])
        segments = t.tensor([
            [[-1.0, 0.0], [-2.0, 0.0]],
            [[ 0.0, 0.0], [ 0.0, 0.0]],
            [[ 1.0, 0.0], [ 2.0, 0.0]],
        ])
        result = intersect_rays_1d(rays, segments)
        self.assertFalse(t.allclose(result, t.tensor([True, True])))

if MAIN:
    unittest.main()
# %%

# TODO
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    # SOLUTION
    n_pixels = num_pixels_y * num_pixels_z
    ygrid = t.linspace(-y_limit, y_limit, num_pixels_y)
    zgrid = t.linspace(-z_limit, z_limit, num_pixels_z)
    rays = t.zeros((n_pixels, 2, 3), dtype=t.float32)
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)
    return rays


if MAIN:
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    render_lines_with_plotly(rays_2d)
# %%


if MAIN:
    one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
    A, B, C = one_triangle
    x, y, z = one_triangle.T

	# TODO
    # fig = setup_widget_fig_triangle(x, y, z)

# @interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
# def response(u=0.0, v=0.0):
#     P = A + u * (B - A) + v * (C - A)
#     fig.data[2].update({"x": [P[0]], "y": [P[1]]})


if MAIN:
    pass
    # TODO
    # display(fig)
# %%

Point = Float[Tensor, "points=3"]

@jaxtyped
@typeguard.typechecked
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''
    # TODO
    s, u, v = t.linalg.solve(
        t.stack([-D, B - A, C - A], dim=1), 
        O - A
    )
    return ((u >= 0) & (v >= 0) & (u + v <= 1)).item()


if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%

if MAIN:
    x = t.Tensor([1])
    y = t.Tensor([1])
    print(x.storage().data_ptr())
    print(y.storage().data_ptr())
    print(x.storage().data_ptr() == y.storage().data_ptr())

    x = t.zeros(1024)
    y = x[0]
    print(x.storage().data_ptr())
    del x
    print(y.storage().data_ptr())
    print(y._base.storage().data_ptr())
# %%

def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    # TODO
    # SOLUTION
    NR = rays.size(0)

    # Triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)
    assert A.shape == (NR, 3)

    # Each element of `rays` is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    O, D = rays.unbind(dim=1)
    assert O.shape == (NR, 3)

    # Define matrix on left hand side of equation
    mat: Float[Tensor, "NR 3 3"] = t.stack([- D, B - A, C - A], dim=-1)

    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    # Note - this works because mat[is_singular] has shape (NR_where_singular, 3, 3), so we
    # can broadcast the identity matrix to that shape.
    dets: Float[Tensor, "NR"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec = O - A

    # Solve eqns
    sol: Float[Tensor, "NR 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    # Return boolean of (matrix is nonsingular, and solution is in correct range implying intersection)
    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


if MAIN:
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 15
    y_limit = z_limit = 0.5

    # Plot triangle & rays
    test_triangle = t.stack([A, B, C], dim=0)
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
    render_lines_with_plotly(rays2d, triangle_lines)

    # Calculate and display intersections
    intersects = raytrace_triangle(rays2d, test_triangle)
    img = intersects.reshape(num_pixels_y, num_pixels_z).int()
    imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%

# TODO
def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size[0]

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(-1)

    mat = t.stack([- D, B - A, C - A])

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%
if MAIN:
    with open(section_dir / "pikachu.pt", "rb") as f:
        triangles = t.load(f)
# %%

def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    # TODO

    # SOLUTION
    NR = rays.size(0)
    NT = triangles.size(0)

    # Each triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    triangles = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)
    A, B, C = triangles
    assert A.shape == (NR, NT, 3)

    # Each ray is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    rays = einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)
    O, D = rays
    assert O.shape == (NR, NT, 3)

    # Define matrix on left hand side of equation
    mat: Float[Tensor, "NR NT 3 3"] = t.stack([- D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets: Float[Tensor, "NR NT"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec: Float[Tensor, "NR NT 3"] = O - A

    # Solve eqns (note, s is the distance along ray)
    sol: Float[Tensor, "NR NT 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(-1)

    # Get boolean of intersects, and use it to set distance to infinity wherever there is no intersection
    intersects = ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)
    s[~intersects] = float("inf") # t.inf

    # Get the minimum distance (over all triangles) for each ray
    return s.min(dim=-1).values


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()
# %%
