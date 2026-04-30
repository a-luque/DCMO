from dataclasses import dataclass
from typing import Dict
import networkx as nx
import numpy as np
import itertools
from typing import List, Tuple, Dict

@dataclass
class Rule:
    name:        str
    description: str
    affinity:    float   # +1 = pure efficiency, -1 = pure stability

# Rules are behavioral preferences, not hard constraints. 
# TODO
# We call stage1 safety specifications, and it is hard constraint. Does stage2 count as soft constraints?
# Stage3, instead of dynamic rulebook / rules, we can call it preference specifications.

RULES = [
    Rule(
        name="prioritize_speed",
        description=(
            "Prefer higher avg_speed even at cost of smoothness. "
            "Relevant when road is clear and lead car is fast or absent."
        ),
        affinity=+1.0,
    ),
    Rule(
        name="prefer_light_model",
        description=(
            "Favor smaller ResNet (18/50) over larger (101) for efficiency. "
            "When road conditions are simple, lighter model suffices."
        ),
        affinity=+0.8,
    ),
    Rule(
        name="minimize_jerk",
        description=(
            "Keep acceleration changes smooth. "
            "Critical when close to lead car or in adverse weather."
        ),
        affinity=-1.0,
    ),
    Rule(
        name="maintain_following_distance",
        description=(
            "Preserve buffer to lead car. "
            "When dist is small or lead is slow, smooth braking matters more."
        ),
        affinity=-0.8,
    ),
    Rule(
        name="weather_caution",
        description=(
            "In HardRain or ClearSunset, prefer stability. "
            "Reduced grip/visibility makes jerk more dangerous."
        ),
        affinity=-0.9,
    ),
]

RULE_NAMES    = [r.name for r in RULES]
RULE_AFFINITY = {r.name: r.affinity for r in RULES}
def get_priority_edges(context: tuple) -> list:
    weather, dist, lead_speed = context
    no_lead = (dist == 100 and lead_speed == 0)
    edges   = []

    if no_lead:
        edges += [
            ("prioritize_speed",   "minimize_jerk"),
            ("prioritize_speed",   "maintain_following_distance"),
            ("prefer_light_model", "minimize_jerk"),
            # REMOVED: ("weather_caution", "prioritize_speed")
            # weather block below handles this conditionally
        ]

    else:
        closing_risk  = (dist <= 5)
        building_risk = (dist <= 10 and lead_speed <= 6)
        low_risk      = (dist >= 15 or lead_speed >= 8)

        if closing_risk:
            edges += [
                ("minimize_jerk",              "prioritize_speed"),
                ("minimize_jerk",              "prefer_light_model"),
                ("maintain_following_distance","prioritize_speed"),
                ("maintain_following_distance","prefer_light_model"),
            ]
            if lead_speed <= 4:
                edges += [
                    ("maintain_following_distance", "minimize_jerk"),
                ]

        elif building_risk:
            edges += [
                ("minimize_jerk",               "prioritize_speed"),
                ("maintain_following_distance",  "prefer_light_model"),
                ("maintain_following_distance",  "prioritize_speed"),
            ]

        elif low_risk:
            edges += [
                ("prioritize_speed",            "minimize_jerk"),
                ("prefer_light_model",          "minimize_jerk"),
                ("maintain_following_distance", "prioritize_speed"),
            ]

        if lead_speed <= 4:
            edges += [
                ("minimize_jerk",               "prioritize_speed"),
                ("maintain_following_distance", "prefer_light_model"),
            ]
        elif lead_speed >= 8:
            edges += [
                ("prioritize_speed",   "maintain_following_distance"),
                ("prefer_light_model", "minimize_jerk"),
            ]

    # ------------------------------------------------------------------
    # WEATHER — now correctly only adds edges when weather warrants it
    # ------------------------------------------------------------------
    if weather == 'HardRainNoon':
        # Rain: stability dominates in ALL contexts including open road
        edges += [
            ("weather_caution",   "prioritize_speed"),
            ("weather_caution",   "prefer_light_model"),
            ("minimize_jerk",     "prioritize_speed"),
            ("minimize_jerk",     "prefer_light_model"),
        ]
        if not no_lead and dist <= 10:
            edges += [
                ("maintain_following_distance", "prioritize_speed"),
                ("weather_caution",             "prefer_light_model"),
            ]

    elif weather == 'ClearSunset':
        if no_lead:
            # Open road at sunset: mild caution but NOT enough to flip
            # Only add ONE mild edge rather than two stability edges
            edges += [
                ("weather_caution", "minimize_jerk"),  
                # weather_caution above minimize_jerk, not above speed
            ]
        else:
            # Lead car present at sunset: caution more justified
            edges += [
                ("weather_caution",  "prioritize_speed"),
                ("minimize_jerk",    "prefer_light_model"),
            ]

    # ClearNoon: no weather edges at all

    return list(set(edges))


def build_priority_graph(edges: list, rule_names: list) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(rule_names)
    G.add_edges_from(edges)
    return G


def graph_to_objective_weights(
    G:             nx.DiGraph,
    rule_names:    List[str],
    rule_affinity: Dict[str, float],
) -> Tuple[float, float]:
    """
    1. Compute per-rule importance via 2^m scheme (paper Definition 6)
    2. Multiply by affinity → signed contribution
    3. Aggregate into (w_efficiency, w_stability)
    """
    # Step 1: 2^m importance per rule
    tc = nx.transitive_closure(G)
    rule_importance = {}
    for rule in rule_names:
        m = len(list(tc.successors(rule)))   # rules this rule dominates
        rule_importance[rule] = 2 ** m

    # Step 2 & 3: aggregate by affinity sign
    eff_score  = 0.0
    stab_score = 0.0
    for rule in rule_names:
        importance = rule_importance[rule]
        affinity   = rule_affinity[rule]
        if affinity > 0:
            eff_score  += importance * affinity
        else:
            stab_score += importance * abs(affinity)

    # Normalize
    total = eff_score + stab_score
    if total < 1e-8:
        return (0.5, 0.5)

    return (eff_score / total, stab_score / total)


def build_objective_weight_table(
    contexts:      List[Tuple],
    rule_names:    List[str],
    rule_affinity: Dict[str, float],
) -> Dict[Tuple, Tuple[float, float]]:

    table = {}
    for ctx in contexts:
        edges      = get_priority_edges(ctx)
        G          = build_priority_graph(edges, rule_names)
        w_eff, w_stab = graph_to_objective_weights(G, rule_names, rule_affinity)
        table[ctx] = (w_eff, w_stab)
    return table


# --- Build contexts ---
weather = ['ClearNoon', 'HardRainNoon', 'ClearSunset']
dists   = [5, 10, 15]
speeds  = [4, 6, 8]
contexts = (
    list(itertools.product(weather, dists, speeds)) +
    list(itertools.product(weather, [100], [0]))
)

# --- Build table ---
obj_weight_table = build_objective_weight_table(
    contexts, RULE_NAMES, RULE_AFFINITY
)

"""
# --- Print ---
def print_weight_table(table: dict):
    print(f"\n{'Context':<40} {'w_eff':>7} {'w_stab':>8}  preference")
    print("─" * 65)

    groups = {
        'ClearNoon':    [],
        'HardRainNoon': [],
        'ClearSunset':  [],
    }
    for ctx, (we, ws) in table.items():
        groups[ctx[0]].append((ctx, we, ws))

    for weather_group, items in groups.items():
        print(f"\n  [{weather_group}]")
        for ctx, we, ws in sorted(items, key=lambda x: (x[0][1], x[0][2])):
            pref  = "efficiency ▶" if we > ws else "◀ stability"
            ratio = we / ws if ws > 0 else float('inf')
            print(f"  {str(ctx):<38} {we:>7.3f} {ws:>8.3f}  {pref}  "
                  f"(ratio {ratio:.2f})")


print_weight_table(obj_weight_table)


def sanity_check(table: dict):
    """
    These should all pass given the domain semantics.
    """
    checks = [
        # (context, expected, reason)
        (('HardRainNoon',  5, 4), "stability",
         "Rain + very close + slow lead: hardest braking scenario"),

        (('HardRainNoon', 15, 8), "stability",
         "Rain always favors stability"),

        (('HardRainNoon', 100, 0), "stability",
         "Rain even on open road: grip is reduced"),

        (('ClearNoon',   100, 0), "efficiency",
         "Clear + open road: no conflict, go fast"),

        (('ClearNoon',    15, 8), "efficiency",
         "Clear + far + fast lead: lead pulling away, efficiency viable"),

        (('ClearNoon',     5, 4), "stability",
         "Clear but very close + slow: must brake smoothly"),

        (('ClearSunset', 100, 0), "efficiency",
         "Sunset open road: mild caution but efficiency still wins"),

        (('ClearSunset',   5, 4), "stability",
         "Sunset + close + slow: reduced visibility amplifies risk"),
    ]

    print("\nSanity checks:")
    print("─" * 75)
    all_pass = True
    for ctx, expected, reason in checks:
        we, ws  = table[ctx]
        got     = "efficiency" if we > ws else "stability"
        ok      = got == expected
        all_pass = all_pass and ok
        icon    = "✓" if ok else "✗"
        print(f"  {icon} {str(ctx):<35} expected={expected:<12} "
              f"got={got:<12}")
        if not ok:
            print(f"      ↳ FAIL: {reason}")
            print(f"        w_eff={we:.3f}, w_stab={ws:.3f}")

    print(f"\n  {'All checks passed ✓' if all_pass else 'Some checks failed ✗'}")


sanity_check(obj_weight_table)
"""