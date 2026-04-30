import itertools
import numpy as np
from typing import Callable
from enum import Enum
from scipy.special import expit
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


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
        weather = ['ClearNoon', 'HardRainNoon', 'ClearSunset']
        # dists  = [5, 10, 20, 30, 50]
        dists = [5, 10, 15]
        speeds = [4, 6, 8]
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



class ControllerStats:
    """
    Compact numpy-backed storage for one controller.

    Arrays are indexed by the dense context index from ContextSpace.

    Shapes
    ------
    counts  : (n_cells,)
    mu_hat  : (n_cells, K)
    """

    def __init__(self, ctx: ContextSpace, K: int):
        self.ctx    = ctx
        self.K      = K
        self.counts = np.zeros(ctx.total_cells, dtype=np.int64)
        self.mu_hat = np.zeros((ctx.total_cells, K), dtype=np.float64)

    def get_count(self, cell: tuple) -> int:
        return int(self.counts[self.ctx.index(cell)])

    def get_mu(self, cell: tuple) -> np.ndarray:
        return self.mu_hat[self.ctx.index(cell)]  # shape (K,)

    def update(self, cell: tuple, reward: np.ndarray):
        """Online mean update: mu <- (mu * n + r) / (n + 1)."""
        i = self.ctx.index(cell)
        n = self.counts[i]
        self.mu_hat[i] = (self.mu_hat[i] * n + reward) / (n + 1)
        self.counts[i] = n + 1



class ContextualKObjectiveBandit:
    """
    Contextual K-Objective Monitor Learning bandit (Algorithm 2,
    discretized-context variant).

    Differences from the continuous Hölder version:
    - No Hölder parameters (L, alpha, m) and no non-vanishing term v.
    - beta is compared directly against u_{c_t, xi_t}  (line 15).
    - A_T includes K:  1 + 2*log(K * |C_safe| * |X| * T^{3/2})  (line 9).
    - Candidate set uses  g^k_{c,xi} >= mu_hat^k_{c_t,xi} - u_{c_t,xi}
      with no 2v offset  (line 18).

    Parameters
    ----------
    T             : int, total rounds
    ctx           : ContextSpace
    beta          : float, uncertainty threshold for objective refinement
    eps           : float, safety violation probability threshold
    K             : int, number of objectives
    n_controllers : int, number of controllers
    pi_safe       : np.ndarray, safety monitor weight matrix
    """

    def __init__(
        self,
        T: int,
        ctx: ContextSpace,
        beta: float,
        eps: float,
        K: int,
        n_controllers: int,
        pi_safe,
        checkpoint_path: str = "bandit_checkpoint.npz",
        checkpoint_every: int = 10,
    ):
        self.T       = T
        self.ctx     = ctx
        self.beta    = beta
        self.eps     = eps
        self.K       = K
        self.C       = list(range(n_controllers))
        self.checkpoint_path  = checkpoint_path
        self.checkpoint_every = checkpoint_every
        # Collect all .pt files from ../controllers, sorted for stable indexing
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        controllers_path = os.path.join(current_file_dir, "..", "controllers")

        self.controllers_dir = sorted([
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(controllers_path)
            for f in fn
            if f.endswith(".pt")
        ])
        self.pi_safe = pi_safe

        # Per-controller storage: counts + estimated rewards
        self.stats: dict[object, ControllerStats] = {
            c: ControllerStats(ctx, K) for c in self.C
        }

        self.history: list[dict] = []
        """
        self.history.append({
            "controller_pairs": [
                {"index": idx, "path": path}
                for idx, path in enumerate(self.controllers_dir)
            ]
        })
        """

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def _save_checkpoint(self, t: int):
        """Persist bandit state to disk so training can be resumed after a crash."""
        counts_stack = np.stack([self.stats[c].counts for c in self.C], axis=0)   # (n_ctrl, n_cells)
        mu_stack     = np.stack([self.stats[c].mu_hat for c in self.C], axis=0)   # (n_ctrl, n_cells, K)

        # Serialise history list as a flat object array (one entry per round)
        history_arr = np.empty(len(self.history), dtype=object)
        for i, h in enumerate(self.history):
            history_arr[i] = h

        np.savez(
            self.checkpoint_path,
            t=np.array(t),
            counts=counts_stack,
            mu_hat=mu_stack,
            history=history_arr,
        )
        print(f"[checkpoint] Saved at round {t} -> {self.checkpoint_path}")

    def load_checkpoint(self) -> int:
        """
        Load a previously saved checkpoint.

        Returns the last completed round t so that run() can resume from t+1.
        Returns 0 if no checkpoint exists.
        """
        if not os.path.exists(self.checkpoint_path):
            print("[checkpoint] No checkpoint found — starting from scratch.")
            return 0

        data = np.load(self.checkpoint_path, allow_pickle=True)
        t_done = int(data["t"])

        counts_stack = data["counts"]   # (n_ctrl, n_cells)
        mu_stack     = data["mu_hat"]   # (n_ctrl, n_cells, K)
        for c in self.C:
            self.stats[c].counts = counts_stack[c]
            self.stats[c].mu_hat = mu_stack[c]

        self.history = list(data["history"])

        print(f"[checkpoint] Resumed from round {t_done} ({self.checkpoint_path})")
        return t_done

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _safe_controllers(self):
        """Return C_safe = {c in C : [pi_safe(xi)]_c <= eps}."""
        # BUG FIX: was using undefined 'context' and 'dist'; use the correct unpacked names
        #weather_name, intersection, distance, speed = self.ctx.sampled_ctx
        weather_name, distance, speed = self.ctx.sampled_ctx
        xi = np.concatenate([
            np.array(Weather[weather_name].value),
            #np.array([intersection]),
            np.array([distance]),
            np.array([speed]),
            np.array([1.])
        ])
        violations = expit(np.dot(xi, self.pi_safe.T))
        return [c for c in self.C if violations[c].item() <= self.eps]

    def _A(self, n_safe: int) -> float:
        # A_T := 1 + 2*log(K * |C_safe| * |X| * T^{3/2})
        return 1.0 + 2.0 * np.log(
            self.K * n_safe * self.ctx.total_cells * (self.T ** 1.5)
        )

    def _ucb(self, c, cell: tuple, A: float) -> float:
        n = self.stats[c].get_count(cell)
        return np.inf if n == 0 else np.sqrt(2.0 * A / n)

    def _g(self, c, cell: tuple, u: float) -> np.ndarray:
        """UCB index vector:  g^k_{c,xi} = mu_hat^k_{c,xi} + u_{c,xi}."""
        return self.stats[c].get_mu(cell) + u  # shape (K,)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _decide(self, t: int) -> tuple:
        """
        Make a bandit decision for round t using current stats.
        Returns (t, cell, c_t, u_ct, controller_path).
        Called from the main process so it always reads the latest stats.
        """
        C_safe = self.C

        # Line 5: sample context
        cell = self.ctx.sample_one_context()

        # Line 9: uncertainty levels
        A = self._A(len(C_safe))
        u = {c: self._ucb(c, cell, A) for c in C_safe}

        # Line 10: UCB indices for all objectives
        g = {c: self._g(c, cell, u[c]) for c in C_safe}

        # Line 11: best controller for objective 1 (index 0)
        c_t = max(C_safe, key=lambda c: (g[c][0], np.random.rand()))

        # Lines 12-13: skip refinement if only one safe controller
        if len(C_safe) > 1:
            # Lines 14-19: sequential objective refinement
            for k in range(self.K - 1):                         # k = 1..K-1
                # Line 15: break if uncertainty too high
                if u[c_t] > self.beta:
                    break

                # Line 18: candidate optimal controllers (no 2v offset)
                mu_ct_k = self.stats[c_t].get_mu(cell)[k]
                C_hat_star = [
                    c for c in C_safe
                    if g[c][k] >= mu_ct_k - u[c_t]
                ]

                # Line 19: best controller for next objective
                c_t = max(
                    C_hat_star,
                    key=lambda c: (g[c][k + 1], np.random.rand()),
                )

        controller_path = self.controllers_dir[c_t]
        return (t, cell, c_t, u[c_t], controller_path)

    def run(
        self,
        simulate: Callable[[tuple, str, int], np.ndarray],
        start_t: int = 0,
        n_workers: int = 4,
        carla_ports: list = None,
    ):
        """
        Run Algorithm 2 for T rounds, optionally resuming from start_t.
        Simulations are parallelised across n_workers processes. Decisions
        are made in the main process so stats are updated before each new
        submission, keeping staleness bounded to at most (n_workers - 1) rounds.

        Parameters
        ----------
        simulate  : Callable[[tuple, str, int], np.ndarray]
            Given (cell, controller_path, t), returns (reward_vector, safety_info).
        start_t   : int
            Last completed round (from load_checkpoint). Loop starts at start_t + 1.
        n_workers : int
            Number of parallel Scenic processes. Set to the number of CPU cores
        carla_ports : list of int, optional
            One Carla RPC port per worker (e.g. [2000, 2002, 2004, 2006]).
            If None, all workers connect to port 2000 (single-Carla mode).
            (or GPUs) you can dedicate. Default 4.
        """

        # next_t tracks which round number to submit next
        next_t = start_t + 1

        # futures maps: Future -> (t, cell, c_t, u_ct, controller_path)
        futures: dict = {}

        # Build port list: fall back to all-2000 if none given (single-Carla mode)
        ports = carla_ports if carla_ports else [2000] * n_workers

        def _simulate_with_retry(cell, controller_path, t):
            """Thin wrapper that retries on failure — runs inside worker process."""
            while True:
                try:
                    return simulate(cell, controller_path, t)
                except Exception as e:
                    print(f"[t={t}] Simulation failed: {e}. Retrying in 10s...")
                    time.sleep(10)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,   # called once per worker at startup
            initargs=(ports,),          # passes the port list to the initializer
        ) as executor:

            # --- Seed the pool: submit the first n_workers decisions ---
            while next_t <= self.T and len(futures) < n_workers:
                t, cell, c_t, u_ct, controller_path = self._decide(next_t)
                f = executor.submit(_simulate_with_retry, cell, controller_path, t)
                futures[f] = (t, cell, c_t, u_ct, controller_path)
                print(f"[t={t}] Submitted: controller={c_t}, cell={cell}")
                next_t += 1

            # --- Rolling loop: each completion triggers one new submission ---
            while futures:
                # Block until the next simulation finishes (whichever finishes first)
                done = next(as_completed(futures))
                t_done, cell, c_t, u_ct, controller_path = futures.pop(done)

                try:
                    r_t, safety_info = done.result()
                except Exception as e:
                    # Should not reach here due to retry wrapper, but guard anyway
                    print(f"[t={t_done}] Unrecoverable failure: {e}. Skipping round.")
                    r_t, safety_info = np.zeros(self.K), {}

                # Lines 21-22: update stats immediately so the next decision uses them
                self.stats[c_t].update(cell, r_t)

                self.history.append({
                    "t":               t_done,
                    "cell":            cell,
                    "c_t":             c_t,
                    "u[c_t]":          u_ct,
                    "r_t":             r_t,
                    "controller_path": controller_path,
                    "safety_info":     safety_info,
                })

                print(f"[t={t_done}] Done: controller={c_t}, r={r_t}")

                # Checkpoint (keyed on t_done, not submission order)
                if t_done % self.checkpoint_every == 0:
                    self._save_checkpoint(t_done)

                # Submit the next round now that stats are up to date
                if next_t <= self.T:
                    t, cell, c_t, u_ct, controller_path = self._decide(next_t)
                    f = executor.submit(_simulate_with_retry, cell, controller_path, t)
                    futures[f] = (t, cell, c_t, u_ct, controller_path)
                    print(f"[t={next_t}] Submitted: controller={c_t}, cell={cell}")
                    next_t += 1

        return self.history                                          # line 24


def get_reward(results_path: str, controller_path: str, timestep: float = 0.1,
               tau: float = 1.0) -> list:
    # --- efficiency reward: avg_speed / model_size ---
    # Extract model size (18, 50, or 101) from controller filename
    # e.g. "../controllers/resnet101_coarse_best.pt" -> 101
    controller_basename = os.path.basename(controller_path)
    model_size = None
    for candidate in [18, 50, 101]:
        if f"resnet{candidate}" in controller_basename:
            model_size = candidate
            break
    if model_size is None:
        raise ValueError(
            f"Cannot determine model size from controller path: {controller_path!r}. "
            f"Expected filename to contain 'resnet18', 'resnet50', or 'resnet101'."
        )

    speed_results = np.load(results_path + "speed.npz")
    acc_results   = np.load(results_path + "acc.npz")

    avg_speed = np.mean(speed_results["values"])
    acc       = np.array(acc_results["values"])
    jerks     = np.diff(acc) / timestep
    jerks_mean = np.mean(np.abs(jerks))

    #print(f"Jerk Mean: {np.mean(np.abs(jerks))}, Max: {np.max(np.abs(jerks))}, Min: {np.min(np.abs(jerks))}")

    # Min-max normalization for reward_e
    # Max possible value: fastest speed (8) / smallest model (18)  -> 8/18
    # Min possible value: 0 (absolute minimum)
    #RAW_MAX = 8.0 / 18.0
    RAW_MAX = 8.0 / np.log(19.0)
    RAW_MIN = 0.0
    # raw_e   = avg_speed / model_size
    raw_e = avg_speed / np.log(model_size + 1)
    reward_e = (raw_e - RAW_MIN) / (RAW_MAX - RAW_MIN)     # efficiency (normalized)
    reward_e = float(np.clip(reward_e, 0.0, 1.0))          # guard against edge cases

    # reward_c = np.mean(np.exp(-np.abs(jerks) / tau))       # comfort
    #reward_c = np.mean(np.exp(-(jerks / tau)**2))
    reward_s = np.exp(- ( jerks_mean / tau))

    #### logging safety violation info
    dist_vals = np.load(results_path + "dist.npz")
    cte_vals = np.load(results_path + "true_cte.npz")
    collision_vals = np.load(results_path + "collision.npz")

    min_dist = float(dist_vals["values"].min())-4.6
    lane_invasion = int((np.absolute(cte_vals["values"])>0.7).sum())
    collision_happened = int(collision_vals["values"].max())

    # TODO: Change safe distance value to something proportional to the speed?
    if min_dist < 1 or lane_invasion > 30 or collision_happened:
        safety_violation = 1
    else: 
        safety_violation = 0
    
    safety_info = {
        "min_dist": min_dist,
        "avg_speed": avg_speed,
        "avg_jerks": jerks_mean,
        "lane_invasion": lane_invasion,
        "collision_happened": collision_happened,
        "safety_violation": safety_violation
    }

    return [reward_e, reward_s, safety_info]


# ---------------------------------------------------------------------------
# Per-worker Carla port assignment
# ---------------------------------------------------------------------------
# Each worker process stores its assigned Carla port in this module-level
# global. It is set once at process startup via the ProcessPoolExecutor
# initializer, and then read by every simulate() call in that process.
_worker_carla_port: int = 2000  # default; overwritten by _worker_init

def _worker_init(ports: list):
    """
    Called once per worker process when the pool starts.
    Uses the worker PID to pick a unique slot in the port list.
    """
    global _worker_carla_port
    _worker_carla_port = ports[os.getpid() % len(ports)]
    print(f"[worker pid={os.getpid()}] assigned Carla port {_worker_carla_port}",
          flush=True)


def simulate(cell, controller_path: str, t: int) -> np.ndarray:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir = os.path.join(current_file_dir, f"sim_results_parallel_es/{t}/")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    sampled_weather, sampled_distance, sampled_speed = cell

    scenic_file_path = os.path.join(current_file_dir, "gen_sim_reduced.scenic")

    # Read the Carla port assigned to this worker process by _worker_init
    carla_port = _worker_carla_port

    os.system(
        f"scenic -S {scenic_file_path} --count 1 --time 200 --2d "
        f"--param results_path {results_dir} "
        f"--param controller_path {controller_path} "
        f"--param weather {sampled_weather} "
        #f"--param intersect {sampled_intersect} "
        f"--param dist_car {sampled_distance} "
        f"--param speed_car {sampled_speed}"
        f"--param carla_port {str(carla_port)}"

    )
    """
    cmd = [
        "scenic",
        "-S", scenic_file_path,
        "--count", "1",
        "--time", "200",
        "--2d",
        "--param", "results_path",    results_dir,
        "--param", "controller_path", controller_path,
        "--param", "weather",         str(sampled_weather),
        "--param", "dist_car",        str(sampled_distance),
        "--param", "speed_car",       str(sampled_speed),
        "--param", "carla_port",      str(carla_port),
    ]
    subprocess.run(cmd, check=True)
    """

    reward_e, reward_s, safety_info = get_reward(results_dir, controller_path)
    return np.array([reward_e, reward_s]), safety_info


if __name__ == "__main__":
    np.random.seed(42)

    # BUG FIX: was creating ctx_space but then using undefined 'ctx' everywhere
    ctx = ContextSpace()
    # print(f"First 5 cells: {ctx.cells[:5]}")

    T, K           = 5000, 2
    beta, eps      = 4.5, 0.3
    n_controllers  = 9

    pi_path = "/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/safety_monitor_reduced_1/weights_1000.npy"
    with open(pi_path, "rb") as f:
        pi = np.load(f)

    bandit = ContextualKObjectiveBandit(
        T=T, ctx=ctx, beta=beta, eps=eps, K=K,
        n_controllers=n_controllers, pi_safe=pi,
        checkpoint_path="bandit_checkpoint_parallel_es.npz",
        checkpoint_every=10,         # save every 10 rounds
    )

    # Resume from checkpoint if one exists
    start_t = bandit.load_checkpoint()

    # BUG FIX: context_stream removed — sampling is now internal to bandit.run()
    carla_ports = [2000, 2002, 2004, 2006]   # one port per Carla instance / GPU
    history = bandit.run(simulate, start_t=start_t, n_workers=4, carla_ports=carla_ports)

    played = [h for h in history]

    # BUG FIX: 'controllers' was undefined — use bandit.C
    controllers = bandit.C
    counts = {c: sum(1 for h in played if h["c_t"] == c) for c in controllers}
    avg_r  = {
        c: np.mean([h["r_t"] for h in played if h["c_t"] == c], axis=0)
           if counts[c] > 0 else np.zeros(K)
        for c in controllers
    }

    print("=" * 55)
    print(f"Total rounds : {T}")
    print(f"Safety eps   : {eps}")
    print()
    print(f"{'Controller':<12} {'#Plays':>8}  {'Avg R^1':>8}  {'Avg R^2':>8}")
    print("-" * 55)
    for c in controllers:
        r = avg_r[c]
        print(f"{c:<12} {counts[c]:>8}  {r[0]:>8.3f}  {r[1]:>8.3f}")
    print("=" * 55)

    print("\nController 0 — counts shape:", bandit.stats[0].counts.shape)
    print("Controller 0 — mu_hat shape: ", bandit.stats[0].mu_hat.shape)
    cell0 = ctx.cells[0]
    print(f"Stats for con-0 at {cell0}: mu={bandit.stats[0].get_mu(cell0)}, "
          f"n={bandit.stats[0].get_count(cell0)}")

    # Print final rewards matrix (mu_hat) for all controllers
    print("\n--- Final mu_hat (rewards matrix) per controller ---")
    for c in controllers:
        print(f"Controller {c}: mu_hat shape {bandit.stats[c].mu_hat.shape}")
        print(bandit.stats[c].mu_hat)