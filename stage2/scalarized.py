import itertools
import numpy as np
from typing import Callable
from enum import Enum
from scipy.special import expit
import os
import shutil
import time


class Weather(Enum):
    ClearNoon     = [5.0, 0.0, 0.0, 10.0, -1.0, 45.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    CloudyNoon    = [60.0, 0.0, 0.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetNoon       = [5.0, 0.0, 50.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetCloudyNoon = [60.0, 0.0, 50.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    MidRainyNoon  = [60.0, 60.0, 60.0, 60.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    HardRainNoon  = [100.0, 100.0, 90.0, 100.0, -1.0, 45.0, 7.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    SoftRainNoon  = [20.0, 30.0, 50.0, 30.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    ClearSunset   = [5.0, 0.0, 0.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    CloudySunset  = [60.0, 0.0, 0.0, 10.0, -1.0, 15.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetSunset     = [5.0, 0.0, 50.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    WetCloudySunset = [60.0, 0.0, 50.0, 10.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    MidRainSunset = [60.0, 60.0, 60.0, 60.0, -1.0, 15.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    HardRainSunset = [100.0, 100.0, 90.0, 100.0, -1.0, 15.0, 7.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]
    SoftRainSunset = [20.0, 30.0, 50.0, 30.0, -1.0, 15.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0]


# ---------------------------------------------------------------------------
# ContextSpace and ControllerStats are identical to alg_log_es.py
# ---------------------------------------------------------------------------

class ContextSpace:
    """
    Identical to alg_log_es.py — shared between both algorithms so that
    context indexing is exactly the same.
    """
    DIM_NAMES = ("weather", "distance", "speed")

    def __init__(self):
        weather = ['ClearNoon', 'HardRainNoon', 'ClearSunset']
        dists   = [5, 10, 15]
        speeds  = [4, 6, 8]
        self.sampled_ctx = None

        raw = (
            list(itertools.product(weather, dists, speeds)) +
            list(itertools.product(weather, [100], [0]))
        )

        seen = set()
        self.cells = []
        for c in raw:
            if c not in seen:
                seen.add(c)
                self.cells.append(c)

        self._cell_to_idx: dict[tuple, int] = {c: i for i, c in enumerate(self.cells)}

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def total_cells(self) -> int:
        return len(self.cells)

    def index(self, cell: tuple) -> int:
        try:
            return self._cell_to_idx[cell]
        except KeyError:
            raise ValueError(f"Context {cell} is not a valid cell.")

    def cell(self, idx: int) -> tuple:
        return self.cells[idx]

    def sample_one_context(self) -> tuple:
        cell_idx = np.random.randint(self.total_cells)
        self.sampled_ctx = self.cells[cell_idx]
        return self.sampled_ctx

    def __repr__(self):
        return f"ContextSpace(n_cells={self.total_cells}, dims={self.DIM_NAMES})"


class ControllerStats:
    """
    Identical to alg_log_es.py — same storage layout so that mu_hat
    arrays are directly comparable after training.
    """
    def __init__(self, ctx: ContextSpace, K: int):
        self.ctx    = ctx
        self.K      = K
        self.counts = np.zeros(ctx.total_cells, dtype=np.int64)
        self.mu_hat = np.zeros((ctx.total_cells, K), dtype=np.float64)

    def get_count(self, cell: tuple) -> int:
        return int(self.counts[self.ctx.index(cell)])

    def get_mu(self, cell: tuple) -> np.ndarray:
        return self.mu_hat[self.ctx.index(cell)]

    def update(self, cell: tuple, reward: np.ndarray):
        """Online mean update — identical to alg_log_es.py."""
        i = self.ctx.index(cell)
        n = self.counts[i]
        self.mu_hat[i] = (self.mu_hat[i] * n + reward) / (n + 1)
        self.counts[i] = n + 1


# ---------------------------------------------------------------------------
# Scalarized UCB bandit
# ---------------------------------------------------------------------------

class ScalarizedUCBBandit:
    """
    Scalarized multi-objective contextual bandit baseline.

    KEY DIFFERENCE from ContextualKObjectiveBandit (Algorithm 2):
    ---------------------------------------------------------------
    Instead of lexicographic / dominant-objective selection, at every
    round we compute a single scalar UCB index per controller:

        g_{c,xi} = lambda · mu_hat_{c,xi} + u_{c,xi}
                 = sum_k  lambda_k * mu_hat^k_{c,xi}  +  u_{c,xi}

    where  u_{c,xi} = sqrt(2 * A_T / N_{c,xi})  is the same uncertainty
    bonus as in Algorithm 2 (so exploration is directly comparable), and
    lambda in R^K are the fixed scalarization weights (sum to 1).

    Everything else — ContextSpace, ControllerStats, checkpoint / snapshot
    format, simulate() interface, safety filter — is identical to
    alg_log_es.py so that the two algorithms produce mu_hat tables in the
    same shape and can be compared directly.

    Parameters
    ----------
    T             : int   — total rounds
    ctx           : ContextSpace
    lambda_weights: array-like, shape (K,), must sum to 1
    eps           : float — safety violation probability threshold
    K             : int   — number of objectives
    n_controllers : int
    pi_safe       : np.ndarray — safety monitor weight matrix (same as alg_log_es)
    checkpoint_path, checkpoint_every, snapshot_every — same semantics
    """

    def __init__(
        self,
        T: int,
        ctx: ContextSpace,
        lambda_weights,           # <-- the only new parameter vs Algorithm 2
        eps: float,
        K: int,
        n_controllers: int,
        pi_safe,
        checkpoint_path: str  = "bandit_scalar_checkpoint.npz",
        checkpoint_every: int = 10,
        snapshot_every: int   = 1000,
    ):
        self.T              = T
        self.ctx            = ctx
        self.lambda_weights = np.asarray(lambda_weights, dtype=np.float64)
        assert len(self.lambda_weights) == K, "lambda_weights must have length K"
        self.eps            = eps
        self.K              = K
        self.C              = list(range(n_controllers))
        self.checkpoint_path  = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.snapshot_every   = snapshot_every

        base = os.path.splitext(checkpoint_path)[0]
        self.snapshot_dir = base + "_snapshots"
        self.sim_directory = base

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        controllers_path = os.path.join(current_file_dir, "..", "controllers")
        self.controllers_dir = sorted([
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(controllers_path)
            for f in fn
            if f.endswith(".pt")
        ])

        self.pi_safe = pi_safe

        self.stats: dict[int, ControllerStats] = {
            c: ControllerStats(ctx, K) for c in self.C
        }
        self.history: list[dict] = []

    def _save_checkpoint(self, t: int):
        counts_stack = np.stack([self.stats[c].counts for c in self.C], axis=0)
        mu_stack     = np.stack([self.stats[c].mu_hat for c in self.C], axis=0)
        history_arr  = np.empty(len(self.history), dtype=object)
        for i, h in enumerate(self.history):
            history_arr[i] = h
        np.savez(
            self.checkpoint_path,
            t=np.array(t),
            counts=counts_stack,
            mu_hat=mu_stack,
            cells=self.ctx._cell_to_idx,
            history=history_arr,
        )
        print(f"[checkpoint] Saved at round {t} -> {self.checkpoint_path}")

    def _save_snapshot(self, t: int):
        os.makedirs(self.snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(self.snapshot_dir, f"snapshot_t{t}.npz")
        counts_stack  = np.stack([self.stats[c].counts for c in self.C], axis=0)
        mu_stack      = np.stack([self.stats[c].mu_hat for c in self.C], axis=0)
        np.savez(snapshot_path, t=np.array(t), counts=counts_stack, mu_hat=mu_stack)
        print(f"[snapshot]   Saved permanent snapshot at round {t} -> {snapshot_path}")

    def load_checkpoint(self) -> int:
        if not os.path.exists(self.checkpoint_path):
            print("[checkpoint] No checkpoint found — starting from scratch.")
            return 0
        data   = np.load(self.checkpoint_path, allow_pickle=True)
        t_done = int(data["t"])
        counts_stack = data["counts"]
        mu_stack     = data["mu_hat"]
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
        """Identical safety filter to alg_log_es.py."""
        weather_name, distance, speed = self.ctx.sampled_ctx
        xi = np.concatenate([
            np.array(Weather[weather_name].value),
            np.array([distance]),
            np.array([speed]),
            np.array([1.])
        ])
        violations = expit(np.dot(xi, self.pi_safe.T))
        return [c for c in self.C if violations[c].item() <= self.eps]

    def _A(self, n_safe: int) -> float:
        """
        Identical to alg_log_es.py:
            A_T = 1 + 2 * log(K * |C_safe| * |X| * T^{3/2})
        """
        return 1.0 + 2.0 * np.log(
            self.K * n_safe * self.ctx.total_cells * (self.T ** 1.5)
        )

    def _ucb(self, c, cell: tuple, A: float) -> float:
        """Identical uncertainty bonus to alg_log_es.py."""
        n = self.stats[c].get_count(cell)
        return np.inf if n == 0 else np.sqrt(2.0 * A / n)

    def _scalarized_index(self, c, cell: tuple, u: float) -> float:
        """
        THE ONLY DIFFERENCE FROM ALGORITHM 2.

        Algorithm 2 keeps per-objective indices g^k and uses them in a
        two-stage lexicographic selection.

        Here we collapse all objectives into one scalar:

            g_{c,xi} = lambda · mu_hat_{c,xi}  +  u_{c,xi}

        The uncertainty bonus u is the same formula — exploration pressure
        is identical.  Only the *direction* of exploitation changes.
        """
        mu = self.stats[c].get_mu(cell)          # shape (K,)
        scalarized_mu = float(np.dot(self.lambda_weights, mu))
        return scalarized_mu + u

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        simulate: Callable[[tuple, str, str, int], np.ndarray],
        start_t: int = 0,
    ):
        """
        Run for T rounds.  Interface identical to alg_log_es.py so the
        same simulate() function can be reused without modification.
        """
        C_safe = self.C          # safety filter disabled, mirroring alg_log_es.py

        for t in range(start_t + 1, self.T + 1):

            cell = self.ctx.sample_one_context()

            # Uncertainty level — same formula as Algorithm 2
            A = self._A(len(C_safe))
            u = {c: self._ucb(c, cell, A) for c in C_safe}

            # -------------------------------------------------------
            # Controller selection — THIS IS WHERE THE TWO ALGORITHMS DIFFER
            #
            # Algorithm 2:  two-stage lexicographic selection
            #     1. pick c_t = argmax g^1_{c,xi}
            #     2. if u[c_t] <= beta: refine via candidate set on g^2
            #
            # Scalarized:   single-stage, one weighted index
            #     pick c_t = argmax  lambda · mu_hat_{c,xi} + u_{c,xi}
            # -------------------------------------------------------
            g_scalar = {c: self._scalarized_index(c, cell, u[c]) for c in C_safe}
            c_t = max(C_safe, key=lambda c: (g_scalar[c], np.random.rand()))

            controller_path = self.controllers_dir[c_t]

            # Simulate — identical to alg_log_es.py
            while True:
                try:
                    r_t, safety_info = simulate(cell, controller_path, self.sim_directory, t)
                    break
                except Exception as e:
                    print("Simulation failed:", e)
                    time.sleep(5)

            # Update estimates — identical to alg_log_es.py
            self.stats[c_t].update(cell, r_t)

            self.history.append({
                "t":               t,
                "cell":            cell,
                "c_t":             c_t,
                "u[c_t]":          u[c_t],
                "r_t":             r_t,
                "controller_path": controller_path,
                "safety_info":     safety_info,
            })

            if t % self.checkpoint_every == 0:
                self._save_checkpoint(t)
            if t % self.snapshot_every == 0:
                self._save_snapshot(t)

        return self.history


    def get_mu_hat_array(self) -> np.ndarray:
        """
        Returns mu_hat as shape (n_controllers, n_cells, K) — identical
        layout to the npz saved by alg_log_es.py, so comparison code
        can load both with the same indexing.
        """
        return np.stack([self.stats[c].mu_hat for c in self.C], axis=0)

    def get_pareto_front(self, cell: tuple) -> list[tuple[int, np.ndarray]]:
        """
        Post-training: extract Pareto front for a given context cell.
        Returns list of (controller_index, reward_vector) for all
        non-dominated controllers.

        Identical method to what you would call on your Algorithm 2
        mu_hat table — so the two fronts are directly comparable.
        """
        reward_vectors = {c: self.stats[c].get_mu(cell) for c in self.C}

        pareto_front = []
        for c, r in reward_vectors.items():
            dominated = any(
                np.all(r2 >= r) and np.any(r2 > r)
                for c2, r2 in reward_vectors.items() if c2 != c
            )
            if not dominated:
                pareto_front.append((c, r.copy()))

        return pareto_front



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

    # to controll the priorities of objectives, we only need to change the order of reward.
    return [reward_e, reward_s, safety_info] # # objective order: e, s
    #return [reward_s, reward_e, safety_info] # objective order: s, e


def simulate(cell, controller_path: str, file_name: str, t: int) -> np.ndarray:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    #results_dir = f"sim_results/{t}/"
    results_dir = os.path.join(current_file_dir, f"scalarized_sim_results/{file_name}/{t}/") # for order: e,s
    # results_dir = os.path.join(current_file_dir, f"sim_results_se/{t}/") # for order: s, e
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    # sampled_weather, sampled_intersect, sampled_distance, sampled_speed = cell
    sampled_weather, sampled_distance, sampled_speed = cell
    
    scenic_file_path = os.path.join(current_file_dir, "gen_sim_reduced.scenic")
    
    os.system(
        f"scenic -S {scenic_file_path} --count 1 --time 200 --2d "
        f"--param results_path {results_dir} "
        f"--param controller_path {controller_path} "
        f"--param weather {sampled_weather} "
        #f"--param intersect {sampled_intersect} "
        f"--param dist_car {sampled_distance} "
        f"--param speed_car {sampled_speed}"
    )
    
    
    #rewards = get_reward(results_dir, controller_path)
    [reward_e, reward_s, safety_info] = get_reward(results_dir, controller_path)
    rewards = [reward_e, reward_s]
    return np.array(rewards), safety_info



if __name__ == "__main__":
    np.random.seed(42)

    ctx = ContextSpace()
    T, K          = 5000, 2
    eps           = 0.3
    n_controllers = 9

    print("start1")

    pi_path = "/mimer/NOBACKUP/groups/naiss2024-22-1336/DCMO_alj/safety_monitor_reduced_1/weights_1000.npy"
    with open(pi_path, "rb") as f:
        pi = np.load(f)

    # --- Run one scalarized instance per lambda ----------------------------
    # Using three representative weight vectors.
    # lambda = [1, 0]: pure objective-1 (efficiency) — closest to Algorithm 2
    # lambda = [0.5, 0.5]: equal weight
    # lambda = [0, 1]: pure objective-2 (smoothness)
    print("start2")
    lambda_configs = [
        ["lambda_1_0", [1.0, 0.0]],
        ["lambda_07_03", [0.7, 0.3]],
        ["lambda_05_05", [0.5, 0.5]],
        ["lambda_03_07", [0.3, 0.7]],
        ["lambda_0_1", [0.0, 1.0]],
    ]

    lambda_configs_idx = 0
    #for name, lam in lambda_configs.items():
    name = lambda_configs[lambda_configs_idx][0]
    lam = lambda_configs[lambda_configs_idx][1]
    print(f"\n{'='*55}")
    print(f"Training scalarized bandit: {name}  lambda={lam}")
    print(f"{'='*55}")

    bandit = ScalarizedUCBBandit(
        T=T, ctx=ctx,
        lambda_weights=lam,
        eps=eps, K=K,
        n_controllers=n_controllers,
        pi_safe=pi,
        checkpoint_path=f"bandit_scalar_{name}.npz",
        checkpoint_every=10,
        snapshot_every=1000,
    )
    start_t = bandit.load_checkpoint()
    bandit.run(simulate, start_t=start_t)