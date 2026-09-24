import numpy as np
from enum import Enum
import itertools

class Weather(Enum):
    ClearNoon = [5.0, 0.0, 0.0, 10.0, -1.0, 45.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    CloudyNoon = [60.0, 0.0, 0.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetNoon = [5.0, 0.0, 50.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetCloudyNoon = [60.0, 0.0, 50.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    MidRainyNoon = [60.0, 60.0, 60.0, 60.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    HardRainNoon = [100.0, 100.0, 90.0, 100.0, -1.0, 45.0, 7.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    SoftRainNoon = [20.0, 30.0, 50.0, 30.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    ClearSunset = [5.0, 0.0, 0.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    CloudySunset = [60.0, 0.0, 0.0, 10.0, -1.0, 15.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetSunset = [5.0, 0.0, 50.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetCloudySunset = [60.0, 0.0, 50.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    MidRainSunset = [60.0, 60.0, 60.0, 60.0, -1.0, 15.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    HardRainSunset = [100.0, 100.0, 90.0, 100.0, -1.0, 15.0, 7.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    SoftRainSunset = [20.0, 30.0, 50.0, 30.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]


# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]>c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


class ContextSpace:
    """
    Sparse, irregular discretized context space.

    Contexts are explicit (weather, intersection, distance, speed) tuples.
    Internally each valid tuple is mapped to a dense integer index so that
    storage arrays are compact numpy arrays.
    """

    # DIM_NAMES = ("weather", "intersection", "distance", "speed")
    DIM_NAMES = ("weather", "distance", "speed")

    def __init__(self):
        # weather = ['ClearNoon','CloudyNoon','WetNoon','WetCloudyNoon','MidRainyNoon', 'HardRainNoon', 'SoftRainNoon', 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset', 'MidRainSunset', 'HardRainSunset', 'SoftRainSunset']
        #weather = ['ClearNoon', 'HardRainNoon', 'CloudyNoon'] # TODO: foggy noon
        weather = ['ClearNoon']
        # dists  = [5, 10, 20, 30, 50]
        #dists = [5, 10, 15]
        distance_between = 5.0
        dists  = [1, 10, 20, 30] #[6,15), [15,25), [25,35), [35,55)
        dists = [x + distance_between for x in dists]
        speeds = [4, 8, 12]
        self.sampled_ctx = None

        """
            raw = (
                list(itertools.product(weather, [0], dists, speeds)) +  # normal road, finite dist
                list(itertools.product(weather, [1], [5],   speeds)) +  # intersection, close
                list(itertools.product(weather, [0], [100], [0]))    +  # normal road, far/stopped
                list(itertools.product(weather, [1], [100], [0]))       # intersection, far/stopped
            )
        """
        raw = (
                list(itertools.product(weather, dists, speeds)) +  # normal road, finite dist
                list(itertools.product(weather, [100], [0]))      # normal road, far/stopped
            )

        # Deduplicate while preserving order
        seen = set()
        self.cells = []
        for c in raw:
            if c not in seen:
                seen.add(c)
                self.cells.append(c)

        # tuple -> dense index (used once at lookup, not per-round storage)
        self._cell_to_idx: dict[tuple, int] = {c: i for i, c in enumerate(self.cells)}

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def total_cells(self) -> int:
        return len(self.cells)

    def index(self, cell: tuple) -> int:
        """Return the dense index for a context tuple."""
        try:
            return self._cell_to_idx[cell]
        except KeyError:
            raise ValueError(
                f"Context {cell} is not a valid cell. "
                # f"Expected (weather, intersection, distance, speed)."
                f"Expected (weather, distance, speed)."
            )

    def cell(self, idx: int) -> tuple:
        """Return the context tuple for a dense index."""
        return self.cells[idx]

    def sample_one_context(self) -> tuple:
        """Uniform random sample over all valid discrete cells.
            Random index -> context.
        """
        # BUG FIX: was self.ctx.total_cells — ContextSpace IS the context, use self.total_cells
        cell_idx = np.random.randint(self.total_cells)
        self.sampled_ctx = self.cells[cell_idx]
        return self.sampled_ctx

    def __repr__(self):
        return f"ContextSpace(n_cells={self.total_cells}, dims={self.DIM_NAMES})"