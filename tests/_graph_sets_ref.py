"""Confusion graph and maximum independent set (MIS).

Nodes are universe words in frequency order, so the set at any x is a prefix
and every (x, tau) graph is an induced, thresholded subgraph of one edge list.
Solvers: CP-SAT (exact, status checked, deterministic), min-degree greedy with
(1,2) and (1,1)-frequency swaps, and forest (randomized greedy, many seeded
runs, same swaps, best kept).
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class Graph:
    def __init__(self, n: int, ei: np.ndarray, ej: np.ndarray):
        self.n = n
        self.ei, self.ej = ei.astype(np.int64), ej.astype(np.int64)
        self.adj = [set() for _ in range(n)]
        for a, b in zip(self.ei.tolist(), self.ej.tolist()):
            self.adj[a].add(b)
            self.adj[b].add(a)

    @classmethod
    def from_edges(cls, EI, EJ, ED, n_nodes: int, tau: float):
        m = (ED < tau) & (EI < n_nodes) & (EJ < n_nodes)
        return cls(n_nodes, EI[m], EJ[m])

    @classmethod
    def induced(cls, EI, EJ, ED, nodes, tau: float):
        """Subgraph on an arbitrary node subset, reindexed 0..len(nodes)-1 in the
        given order (keep `nodes` sorted to preserve frequency order)."""
        nodes = np.asarray(nodes, np.int64)
        pos = np.full(int(max(EI.max(initial=0), EJ.max(initial=0), nodes.max(initial=0))) + 1,
                      -1, np.int64)
        pos[nodes] = np.arange(len(nodes))
        pi = np.where(EI < len(pos), pos[np.minimum(EI, len(pos) - 1)], -1)
        pj = np.where(EJ < len(pos), pos[np.minimum(EJ, len(pos) - 1)], -1)
        m = (ED < tau) & (pi >= 0) & (pj >= 0)
        return cls(len(nodes), pi[m], pj[m])

    @property
    def n_edges(self):
        return len(self.ei)

    def components(self) -> list[np.ndarray]:
        if self.n == 0:
            return []
        A = coo_matrix((np.ones(len(self.ei)), (self.ei, self.ej)), shape=(self.n, self.n))
        k, lab = connected_components(A, directed=False)
        order = np.argsort(lab, kind="stable")
        splits = np.flatnonzero(np.diff(lab[order])) + 1
        comps = np.split(order, splits)
        return sorted(comps, key=len, reverse=True)

    def is_independent(self, S) -> bool:
        S = set(S)
        return all(not (self.adj[v] & S) for v in S)


# ----------------------------------------------------------------- exact
def _greedy_cliques(nodes, adj):
    """edge-disjoint-ish greedy clique cover: strengthens the CP-SAT model."""
    ns = set(nodes)
    cliques, covered = [], set()
    for v in sorted(nodes, key=lambda x: -len(adj[x] & ns)):
        cand = sorted((adj[v] & ns), key=lambda x: -len(adj[x] & ns))
        c = [v]
        for u in cand:
            if all(u in adj[w] for w in c):
                c.append(u)
        if len(c) >= 3:
            key = frozenset(c)
            if key not in covered:
                covered.add(key)
                cliques.append(c)
    return cliques


def exact_mis(nodes, adj, time_limit: float, workers: int = 8, hint=None,
              weights=None, fix_size=None, seed: int = 0, deterministic: bool = True):
    """CP-SAT on one component.
    weights=None: maximize |S|.  With fix_size=k and weights: maximize sum(w) s.t. |S|=k
    (the lexicographic second phase).
    -> dict(status, value, bound, set, seconds)"""
    from ortools.sat.python import cp_model

    nodes = list(nodes)
    ns = set(nodes)
    m = cp_model.CpModel()
    x = {v: m.NewBoolVar(f"x{v}") for v in nodes}
    for c in _greedy_cliques(nodes, adj):
        m.AddAtMostOne(x[v] for v in c)
    for v in nodes:
        for u in adj[v]:
            if u in ns and u > v:
                m.AddBoolOr([x[v].Not(), x[u].Not()])
    if fix_size is not None:
        m.Add(sum(x.values()) == fix_size)
    if weights is None:
        m.Maximize(sum(x.values()))
    else:
        m.Maximize(sum(int(weights[v]) * x[v] for v in nodes))
    if hint:
        hs = set(hint)
        for v in nodes:
            m.AddHint(x[v], v in hs)
    s = cp_model.CpSolver()
    s.parameters.num_workers = workers
    s.parameters.random_seed = seed
    if deterministic:
        # reproducible: interleaved (deterministic) parallel search, budget in
        # deterministic time units; wall clock is only a safety net
        s.parameters.interleave_search = True
        s.parameters.max_deterministic_time = float(time_limit)
        s.parameters.max_time_in_seconds = float(time_limit) * 3
    else:
        s.parameters.max_time_in_seconds = float(time_limit)
    t0 = time.time()
    st = s.Solve(m)
    secs = time.time() - t0
    name = s.StatusName(st)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        S = [v for v in nodes if s.Value(x[v])]
        val = s.ObjectiveValue()
        bound = s.BestObjectiveBound()
    else:
        S, val, bound = [], float("nan"), float("nan")
    return {"status": name, "value": val, "bound": bound, "set": S, "seconds": secs}


# ----------------------------------------------------------------- heuristics
def mindeg_greedy(nodes, adj):
    ns = set(nodes)
    alive = set(nodes)
    deg = {v: len(adj[v] & ns) for v in nodes}
    h = [(deg[v], v) for v in nodes]
    heapq.heapify(h)
    S = []
    while h:
        d, v = heapq.heappop(h)
        if v not in alive or d != deg[v]:
            continue
        S.append(v)
        alive.discard(v)
        for u in adj[v] & alive:
            alive.discard(u)
            for w in adj[u] & alive:
                deg[w] -= 1
                heapq.heappush(h, (deg[w], w))
    return set(S)


def two_improve(S, nodes, adj, rounds=6):
    """(1,2) swaps: remove v in S, insert two non-adjacent 1-tight neighbours."""
    ns = set(nodes)
    S = set(S)
    for _ in range(rounds):
        gained = 0
        for v in list(S):
            if v not in S:
                continue
            free = [u for u in (adj[v] & ns) if u not in S and len(adj[u] & S) == 1]   # only v blocks u
            done = False
            for i in range(len(free)):
                for j in range(i + 1, len(free)):
                    if free[j] not in adj[free[i]]:
                        S.discard(v)
                        S.add(free[i])
                        S.add(free[j])
                        # any further free vertex independent of the new set also fits
                        for u in free:
                            if u not in S and not (adj[u] & S):
                                S.add(u)
                        gained += 1
                        done = True
                        break
                if done:
                    break
        if gained == 0:
            break
    return S


def freq_swap(S, nodes, adj, rounds=8):
    """(1,1) swaps: same size, replace a rarer member by a more frequent outsider
    (index order == frequency rank, lower = more frequent)."""
    ns = set(nodes)
    S = set(S)
    for _ in range(rounds):
        improved = 0
        for u in sorted(ns - S):
            if u in S:
                continue
            blockers = adj[u] & S
            if len(blockers) == 1:
                v = next(iter(blockers))
                if u < v:
                    S.discard(v)
                    S.add(u)
                    # v's other neighbours may now be free: refill (most frequent first)
                    for w in sorted(adj[v] & ns):
                        if w not in S and not (adj[w] & S):
                            S.add(w)
                    improved += 1
        if improved == 0:
            break
    return S


def greedy_pool(nodes, adj, local_search=True, rounds=6):
    S = mindeg_greedy(nodes, adj)
    if local_search:
        S = two_improve(S, nodes, adj, rounds)
        S = freq_swap(S, nodes, adj)
    return S


def randomized_greedy(nodes, adj, k: int, rng):
    """Min-degree greedy where each step picks uniformly at random among the k
    lowest-(dynamic)-degree remaining nodes instead of always the lowest."""
    ns = set(nodes)
    alive = set(nodes)
    deg = {v: len(adj[v] & ns) for v in nodes}
    h = [(deg[v], v) for v in nodes]
    heapq.heapify(h)
    S = []
    while alive:
        pool = []
        while h and len(pool) < k:
            d, v = heapq.heappop(h)
            if v in alive and d == deg[v] and v not in {p[1] for p in pool}:
                pool.append((d, v))
        if not pool:
            break
        d, v = pool[rng.randint(len(pool))]
        for item in pool:
            if item[1] != v:
                heapq.heappush(h, item)
        S.append(v)
        alive.discard(v)
        for u in adj[v] & alive:
            alive.discard(u)
            for w in adj[u] & alive:
                deg[w] -= 1
                heapq.heappush(h, (deg[w], w))
    return set(S)


def forest(nodes, adj, runs: int, k: int, swap_rounds: int, seed: int):
    """'Forest' search: many randomized greedy trees (seeded), each followed by
    the same (1,2) + (1,1) swaps as greedy; keep the largest (ties: more
    frequent, i.e. smaller index sum)."""
    best = None
    for r in range(runs):
        rng = np.random.RandomState(seed * 100003 + r)
        S = randomized_greedy(nodes, adj, k, rng)
        S = freq_swap(two_improve(S, nodes, adj, swap_rounds), nodes, adj)
        if best is None or len(S) > len(best) or (len(S) == len(best) and sum(S) < sum(best)):
            best = S
    return best


def solve_cell(G: Graph, sv: dict, seed=0):
    """Per (x, tau) cell:
    1. ILP feasibility test on the giant component (CP-SAT, ilp_time_limit_s)
    2. ILP works (OPTIMAL)  -> solve the other components exactly; exact alpha
       ILP doesn't            -> solver upper bound is the reference
    3. greedy (+swaps), forest, and no-swap greedy on every component; gaps are
       measured against exact alpha or the upper bound."""
    t_cell = time.time()
    comps = G.components()
    adj = G.adj
    out = {"n": G.n, "edges": G.n_edges, "components": len(comps),
           "giant": int(len(comps[0])) if comps else 0}
    sets = {"greedy_noswap": [], "greedy": [], "forest": []}
    secs = {"greedy_noswap": 0.0, "greedy": 0.0, "forest": 0.0, "ilp": 0.0}
    ub = 0
    ilp_set = []
    ilp_status = []
    for ci, comp in enumerate(comps):
        nodes = comp.tolist()
        if len(nodes) == 1:
            for k in sets:
                sets[k].append(nodes[0])
            ilp_set.append(nodes[0])
            ub += 1
            continue
        t = time.time()
        g0 = mindeg_greedy(nodes, adj)
        secs["greedy_noswap"] += time.time() - t
        t = time.time()
        g1 = freq_swap(two_improve(g0, nodes, adj, sv["swap_rounds"]), nodes, adj)
        secs["greedy"] += time.time() - t
        t = time.time()
        g2 = forest(nodes, adj, sv["forest_runs"], sv["forest_k"], sv["swap_rounds"], seed)
        secs["forest"] += time.time() - t
        sets["greedy_noswap"] += list(g0)
        sets["greedy"] += list(g1)
        sets["forest"] += list(g2)
        lim = sv["ilp_time_limit_s"] if ci == 0 else sv["component_time_limit_s"]
        hint = g2 if len(g2) >= len(g1) else g1
        r = exact_mis(nodes, adj, lim, sv["workers"], hint=hint, seed=seed,
                      deterministic=sv.get("deterministic", True))
        secs["ilp"] += r["seconds"]
        ilp_status.append(r["status"])
        if ci == 0:
            out["giant_ilp_status"] = r["status"]
            out["giant_ilp_seconds"] = r["seconds"]
        if r["status"] == "OPTIMAL":
            ub += len(r["set"])
            ilp_set += r["set"]
        else:
            ub += int(math.floor(r["bound"] + 1e-9)) if np.isfinite(r["bound"]) else len(nodes)
            ilp_set += r["set"] if len(r["set"]) >= len(hint) else list(hint)
    exact = all(s == "OPTIMAL" for s in ilp_status)
    out["ilp_works"] = out.get("giant_ilp_status", "OPTIMAL") == "OPTIMAL"
    out["alpha_exact"] = exact
    out["reference"] = ub            # exact alpha if alpha_exact else certified upper bound
    best = max([ilp_set, sets["greedy"], sets["forest"]], key=len)
    out["N"] = len(best)
    out["bits"] = math.log2(len(best))
    out["gap_words"] = ub - len(best)
    out["gap_bits"] = math.log2(ub / len(best))
    for k, S in sets.items():
        out[f"N_{k}"] = len(S)
        out[f"gap_{k}_words"] = ub - len(S)
        out[f"gap_{k}_bits"] = math.log2(ub / len(S))
        out[f"sec_{k}"] = secs[k]
    out["sec_ilp"] = secs["ilp"]
    out["sec_cell"] = time.time() - t_cell
    for S in [best] + list(sets.values()):
        assert G.is_independent(S)
    out["_best"] = sorted(int(v) for v in best)
    out["_greedy_noswap"] = sorted(int(v) for v in sets["greedy_noswap"])
    return out


def lexicographic_pool(G: Graph, base_set, cfg_solver, weights, seed=0):
    """Phase 2: among maximum independent sets, prefer frequent words.
    Per component: fix |S| to the phase-1 size, maximize sum(weights)."""
    comps = G.components()
    base = set(base_set)
    out = []
    for ci, comp in enumerate(comps):
        nodes = comp.tolist()
        here = [v for v in nodes if v in base]
        if len(nodes) <= 1:
            out += here
            continue
        r = exact_mis(nodes, G.adj, cfg_solver["component_time_limit_s"],
                      cfg_solver["workers"], hint=here, weights=weights,
                      fix_size=len(here), seed=seed)
        out += r["set"] if r["set"] and len(r["set"]) == len(here) else here
    assert G.is_independent(out)
    return sorted(out)


def caro_wei(G: Graph) -> float:
    deg = np.array([len(a) for a in G.adj])
    return float(np.sum(1.0 / (deg + 1.0)))


def knee(taus, bits):
    """Kneedle (Satopää et al. 2011) on bits(tau), decreasing.  The curve shape
    is decided from the data (mean second difference), not assumed."""
    from kneed import KneeLocator
    taus, bits = np.asarray(taus, float), np.asarray(bits, float)
    if len(taus) < 4:
        return None, None
    curv = "convex" if np.mean(np.diff(bits, 2)) > 0 else "concave"
    kl = KneeLocator(taus, bits, curve=curv, direction="decreasing")
    return (float(kl.knee) if kl.knee is not None else None), curv
