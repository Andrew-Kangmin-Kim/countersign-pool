"""Confusion graph and maximum independent set (MIS).

Nodes are universe words in frequency order, so the set at any x is a prefix
and every (x, tau) graph is an induced, thresholded subgraph of one edge list.
Solvers: CP-SAT (exact, status checked, deterministic), min-degree greedy with
(1,2) and (1,1)-frequency swaps, and forest (randomized greedy, many seeded
runs, same swaps, best kept per component).

The graph is stored as CSR arrays and the heuristics are numba kernels: the
densest cell (x=5%, tau=6) has 19.5 M edges, which Python sets cannot hold in
this machine's memory.  The algorithms are the same as the set version
(tests/_graph_sets_ref.py); components are independent, so running a kernel
on all non-trivial components at once equals running it per component (the
forest keeps the best run separately for every component).
"""
from __future__ import annotations

import heapq
import math
import time

import numba as nb
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class Graph:
    def __init__(self, n: int, ei: np.ndarray, ej: np.ndarray):
        self.n = int(n)
        self.ei = np.asarray(ei, np.int32)
        self.ej = np.asarray(ej, np.int32)
        src = np.concatenate([self.ei, self.ej])
        dst = np.concatenate([self.ej, self.ei])
        order = np.lexsort((dst, src))                 # sorted neighbour lists
        self.indices = dst[order].astype(np.int32)
        self.indptr = np.zeros(self.n + 1, np.int64)
        np.cumsum(np.bincount(src, minlength=self.n), out=self.indptr[1:])
        self._lab = None

    @classmethod
    def from_edges(cls, EI, EJ, ED, n_nodes: int, tau: float):
        m = (ED < tau) & (EI < n_nodes) & (EJ < n_nodes)
        return cls(n_nodes, EI[m], EJ[m])

    @property
    def n_edges(self):
        return len(self.ei)

    def degree(self):
        return np.diff(self.indptr)

    def neighbors(self, v):
        return self.indices[self.indptr[v]:self.indptr[v + 1]]

    def labels(self):
        """component label per node, components numbered by size (0 = giant)"""
        if self._lab is None:
            if self.n == 0:
                self._lab = np.zeros(0, np.int64)
            else:
                A = coo_matrix((np.ones(len(self.ei), np.int8), (self.ei, self.ej)),
                               shape=(self.n, self.n))
                _, lab = connected_components(A, directed=False)
                sizes = np.bincount(lab)
                rank = np.empty_like(sizes)
                rank[np.lexsort((np.arange(len(sizes)), -sizes))] = np.arange(len(sizes))
                self._lab = rank[lab]
        return self._lab

    def components(self) -> list[np.ndarray]:
        lab = self.labels()
        order = np.argsort(lab, kind="stable")
        return np.split(order, np.flatnonzero(np.diff(lab[order])) + 1) if self.n else []

    def mask(self, S):
        m = np.zeros(self.n, np.bool_)
        m[np.asarray(list(S), np.int64)] = True
        return m

    def is_independent(self, S) -> bool:
        m = self.mask(S)
        return not np.any(m[self.ei] & m[self.ej])

    def is_maximal(self, S) -> bool:
        m = self.mask(S)
        blocked = np.zeros(self.n, np.bool_)
        blocked[self.ej[m[self.ei]]] = True
        blocked[self.ei[m[self.ej]]] = True
        return bool(np.all(m | blocked))


# ----------------------------------------------------------------- kernels
@nb.njit(cache=True)
def _adjacent(indptr, indices, a, b):
    lo, hi = indptr[a], indptr[a + 1]
    while lo < hi:
        mid = (lo + hi) // 2
        if indices[mid] < b:
            lo = mid + 1
        else:
            hi = mid
    return lo < indptr[a + 1] and indices[lo] == b


@nb.njit(cache=True)
def _greedy(indptr, indices, active, k, seed):
    """min-degree greedy on the active nodes; k > 1 picks uniformly among the k
    lowest-(dynamic)-degree candidates (randomized greedy).  -> bool mask"""
    n = len(indptr) - 1
    if k > 1:
        np.random.seed(seed)
    alive = active.copy()
    deg = np.zeros(n, np.int64)
    h = [(np.int64(0), np.int64(0))]
    h.pop()
    for v in range(n):
        if active[v]:
            deg[v] = indptr[v + 1] - indptr[v]
            h.append((deg[v], np.int64(v)))
    heapq.heapify(h)
    inS = np.zeros(n, np.bool_)
    pool = [(np.int64(0), np.int64(0))]
    while len(h) > 0:
        pool.clear()
        while len(h) > 0 and len(pool) < k:
            d, v = heapq.heappop(h)
            if alive[v] and d == deg[v]:
                pool.append((d, v))
        if len(pool) == 0:
            break
        pick = 0 if k == 1 else np.random.randint(0, len(pool))
        v = pool[pick][1]
        for i in range(len(pool)):
            if i != pick:
                heapq.heappush(h, pool[i])
        inS[v] = True
        alive[v] = False
        for p in range(indptr[v], indptr[v + 1]):
            u = indices[p]
            if alive[u]:
                alive[u] = False
                for q in range(indptr[u], indptr[u + 1]):
                    w = indices[q]
                    if alive[w]:
                        deg[w] -= 1
                        heapq.heappush(h, (deg[w], np.int64(w)))
    return inS


@nb.njit(cache=True)
def _tight(indptr, indices, inS):
    n = len(indptr) - 1
    cnt = np.zeros(n, np.int64)
    for v in range(n):
        if inS[v]:
            for p in range(indptr[v], indptr[v + 1]):
                cnt[indices[p]] += 1
    return cnt


@nb.njit(cache=True)
def _add(indptr, indices, inS, cnt, v, d):
    inS[v] = d > 0
    for p in range(indptr[v], indptr[v + 1]):
        cnt[indices[p]] += d


@nb.njit(cache=True)
def _two_improve(indptr, indices, inS, rounds):
    """(1,2) swaps: remove v in S, insert two non-adjacent neighbours that only
    v blocks; then any further freed neighbour that fits."""
    n = len(indptr) - 1
    inS = inS.copy()
    cnt = _tight(indptr, indices, inS)
    free = np.empty(n, np.int64)
    for _ in range(rounds):
        gained = 0
        snap = np.flatnonzero(inS)
        for v in snap:
            if not inS[v]:
                continue
            f = 0
            for p in range(indptr[v], indptr[v + 1]):
                u = indices[p]
                if not inS[u] and cnt[u] == 1:
                    free[f] = u
                    f += 1
            found = False
            for i in range(f):
                for j in range(i + 1, f):
                    if not _adjacent(indptr, indices, free[i], free[j]):
                        _add(indptr, indices, inS, cnt, v, -1)
                        _add(indptr, indices, inS, cnt, free[i], 1)
                        _add(indptr, indices, inS, cnt, free[j], 1)
                        for t in range(f):
                            u = free[t]
                            if not inS[u] and cnt[u] == 0:
                                _add(indptr, indices, inS, cnt, u, 1)
                        gained += 1
                        found = True
                        break
                if found:
                    break
        if gained == 0:
            break
    return inS


@nb.njit(cache=True)
def _freq_swap(indptr, indices, inS, active, rounds):
    """(1,1) swaps: same size, replace a rarer member by a more frequent outsider
    (index order == frequency rank, lower = more frequent), then refill."""
    n = len(indptr) - 1
    inS = inS.copy()
    cnt = _tight(indptr, indices, inS)
    for _ in range(rounds):
        improved = 0
        for u in range(n):
            if not active[u] or inS[u] or cnt[u] != 1:
                continue
            v = -1
            for p in range(indptr[u], indptr[u + 1]):
                if inS[indices[p]]:
                    v = indices[p]
                    break
            if u < v:
                _add(indptr, indices, inS, cnt, v, -1)
                _add(indptr, indices, inS, cnt, u, 1)
                for p in range(indptr[v], indptr[v + 1]):
                    w = indices[p]
                    if not inS[w] and cnt[w] == 0:
                        _add(indptr, indices, inS, cnt, w, 1)
                improved += 1
        if improved == 0:
            break
    return inS


@nb.njit(cache=True)
def _forest(indptr, indices, active, lab, n_comp, runs, k, swap_rounds, seed):
    n = len(indptr) - 1
    best = np.zeros(n, np.bool_)
    bsize = np.full(n_comp, -1, np.int64)
    bsum = np.zeros(n_comp, np.int64)
    for r in range(runs):
        S = _greedy(indptr, indices, active, k, seed * 100003 + r)
        S = _two_improve(indptr, indices, S, swap_rounds)
        S = _freq_swap(indptr, indices, S, active, 8)
        size = np.zeros(n_comp, np.int64)
        ssum = np.zeros(n_comp, np.int64)
        for v in range(n):
            if active[v] and S[v]:
                size[lab[v]] += 1
                ssum[lab[v]] += v
        better = (size > bsize) | ((size == bsize) & (ssum < bsum))
        for c in range(n_comp):
            if better[c]:
                bsize[c] = size[c]
                bsum[c] = ssum[c]
        for v in range(n):
            if active[v] and better[lab[v]]:
                best[v] = S[v]
    return best


@nb.njit(cache=True)
def _clique_cover(indptr, indices, nodes, member):
    """greedy cliques (size >= 3) to strengthen the CP-SAT model: each node in
    decreasing degree order grows a clique from its neighbours in decreasing
    degree order.  -> flat array of nodes, offsets"""
    deg = np.zeros(len(nodes), np.int64)
    for i in range(len(nodes)):
        v = nodes[i]
        deg[i] = indptr[v + 1] - indptr[v]
    order = np.argsort(-deg, kind="mergesort")
    out = [np.int64(0)]
    out.pop()
    offs = [np.int64(0)]
    c = np.empty(len(nodes), np.int64)
    for oi in order:
        v = nodes[oi]
        nb_ = indices[indptr[v]:indptr[v + 1]]
        nd = np.empty(len(nb_), np.int64)
        for t in range(len(nb_)):
            nd[t] = -(indptr[nb_[t] + 1] - indptr[nb_[t]])
        cs = 1
        c[0] = v
        for t in np.argsort(nd, kind="mergesort"):
            u = nb_[t]
            if not member[u]:
                continue
            ok = True
            for s in range(cs):
                if not _adjacent(indptr, indices, u, c[s]):
                    ok = False
                    break
            if ok:
                c[cs] = u
                cs += 1
        if cs >= 3:
            for s in range(cs):
                out.append(c[s])
            offs.append(np.int64(len(out)))
    return np.array(out, np.int64), np.array(offs, np.int64)


# ----------------------------------------------------------------- python API
def _active(G: Graph, nodes=None):
    if nodes is None:
        return G.degree() > 0
    m = np.zeros(G.n, np.bool_)
    m[np.asarray(nodes, np.int64)] = True
    return m


def mindeg_greedy(G: Graph, nodes=None) -> np.ndarray:
    return np.flatnonzero(_greedy(G.indptr, G.indices, _active(G, nodes), 1, 0))


def randomized_greedy(G: Graph, nodes=None, k: int = 3, seed: int = 0) -> np.ndarray:
    return np.flatnonzero(_greedy(G.indptr, G.indices, _active(G, nodes), k, seed))


def two_improve(G: Graph, S, rounds=6) -> np.ndarray:
    return np.flatnonzero(_two_improve(G.indptr, G.indices, G.mask(S), rounds))


def freq_swap(G: Graph, S, nodes=None, rounds=8) -> np.ndarray:
    return np.flatnonzero(_freq_swap(G.indptr, G.indices, G.mask(S), _active(G, nodes), rounds))


def forest(G: Graph, nodes=None, runs=100, k=3, swap_rounds=6, seed=0) -> np.ndarray:
    lab = G.labels()
    n_comp = int(lab.max()) + 1 if G.n else 0
    return np.flatnonzero(_forest(G.indptr, G.indices, _active(G, nodes), lab, n_comp,
                                  runs, k, swap_rounds, seed))


def exact_mis(G: Graph, nodes, eu, ev, time_limit: float, workers: int = 8, hint=None,
              weights=None, fix_size=None, seed: int = 0, deterministic: bool = True):
    """CP-SAT on one component (nodes; its edges eu-ev).
    weights=None: maximize |S|.  With fix_size=k and weights: maximize sum(w) s.t. |S|=k
    (the lexicographic second phase).
    -> dict(status, value, bound, set, seconds)"""
    from ortools.sat.python import cp_model

    nodes = np.asarray(nodes, np.int64)
    member = np.zeros(G.n, np.bool_)
    member[nodes] = True
    m = cp_model.CpModel()
    x = {int(v): m.NewBoolVar(f"x{v}") for v in nodes}
    flat, offs = _clique_cover(G.indptr, G.indices, nodes, member)
    seen = set()
    for a, b in zip(offs[:-1], offs[1:]):
        key = tuple(sorted(flat[a:b].tolist()))
        if key not in seen:
            seen.add(key)
            m.AddAtMostOne(x[v] for v in key)
    for a, b in zip(np.asarray(eu).tolist(), np.asarray(ev).tolist()):
        m.AddBoolOr([x[a].Not(), x[b].Not()])
    if fix_size is not None:
        m.Add(sum(x.values()) == fix_size)
    if weights is None:
        m.Maximize(sum(x.values()))
    else:
        m.Maximize(sum(int(weights[v]) * x[v] for v in x))
    if hint is not None:
        hs = set(int(v) for v in hint)
        for v in x:
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
        S = [v for v in x if s.Value(x[v])]
        val = s.ObjectiveValue()
        bound = s.BestObjectiveBound()
    else:
        S, val, bound = [], float("nan"), float("nan")
    return {"status": name, "value": val, "bound": bound, "set": S, "seconds": secs}


def _component_edges(G: Graph):
    """edges grouped by component -> (eu, ev, starts) with edges of component c
    in eu[starts[c]:starts[c+1]]"""
    lab = G.labels()
    n_comp = int(lab.max()) + 1 if G.n else 0
    el = lab[G.ei]
    o = np.argsort(el, kind="stable")
    starts = np.zeros(n_comp + 1, np.int64)
    np.cumsum(np.bincount(el, minlength=n_comp), out=starts[1:])
    return G.ei[o], G.ej[o], starts


def solve_cell(G: Graph, sv: dict, seed=0):
    """Per (x, tau) cell:
    1. ILP feasibility test on the giant component (CP-SAT, ilp_time_limit_s)
    2. ILP works (OPTIMAL)  -> solve the other components exactly; exact alpha
       ILP doesn't            -> solver upper bound is the reference
    3. greedy (+swaps), forest, and no-swap greedy on every component; gaps are
       measured against exact alpha or the upper bound."""
    t_cell = time.time()
    lab = G.labels()
    comps = G.components()
    act = G.degree() > 0
    out = {"n": G.n, "edges": G.n_edges, "components": len(comps),
           "giant": int(len(comps[0])) if comps else 0}
    secs = {}
    t = time.time()
    g0 = _greedy(G.indptr, G.indices, act, 1, 0)
    secs["greedy_noswap"] = time.time() - t
    t = time.time()
    g1 = _freq_swap(G.indptr, G.indices, _two_improve(G.indptr, G.indices, g0, sv["swap_rounds"]),
                    act, 8)
    secs["greedy"] = time.time() - t
    t = time.time()
    n_comp = len(comps)
    g2 = _forest(G.indptr, G.indices, act, lab, n_comp, sv["forest_runs"], sv["forest_k"],
                 sv["swap_rounds"], seed)
    secs["forest"] = time.time() - t
    single = ~act
    sets = {k: np.flatnonzero(m | single) for k, m in
            (("greedy_noswap", g0), ("greedy", g1), ("forest", g2))}

    eu, ev, starts = _component_edges(G)
    ilp = single.copy()
    ub = int(single.sum())
    has_bound = True
    ilp_status = []
    secs["ilp"] = 0.0
    max_e = sv.get("ilp_max_edges")
    for ci, comp in enumerate(comps):
        if len(comp) == 1:
            continue
        hint_mask = g2 if g2[comp].sum() >= g1[comp].sum() else g1
        hint = comp[hint_mask[comp]]
        ne = int(starts[ci + 1] - starts[ci])
        if max_e is not None and ne > max_e:
            # model would not fit in memory: no solver run, no bound
            r = {"status": "NOT_RUN_TOO_LARGE", "seconds": 0.0, "set": [], "bound": float("nan")}
        else:
            lim = sv["ilp_time_limit_s"] if ci == 0 else sv["component_time_limit_s"]
            r = exact_mis(G, comp, eu[starts[ci]:starts[ci + 1]], ev[starts[ci]:starts[ci + 1]],
                          lim, sv["workers"], hint=hint, seed=seed,
                          deterministic=sv.get("deterministic", True))
        secs["ilp"] += r["seconds"]
        ilp_status.append(r["status"])
        if ci == 0:
            out["giant_edges"] = ne
            out["giant_ilp_status"] = r["status"]
            out["giant_ilp_seconds"] = r["seconds"]
        if r["status"] == "OPTIMAL":
            ub += len(r["set"])
            ilp[r["set"]] = True
        else:
            if np.isfinite(r["bound"]):
                ub += int(math.floor(r["bound"] + 1e-9))
            else:
                has_bound = False        # no certified bound for this component
            ilp[r["set"] if len(r["set"]) >= len(hint) else hint] = True
    exact = all(s == "OPTIMAL" for s in ilp_status)
    out["ilp_works"] = out.get("giant_ilp_status", "OPTIMAL") == "OPTIMAL"
    out["alpha_exact"] = exact
    # exact alpha if alpha_exact, else certified upper bound; NaN when some
    # component has no solver bound (gaps are then undefined, not guessed)
    ref = ub if has_bound else float("nan")
    out["reference"] = ref
    best = max([np.flatnonzero(ilp), sets["greedy"], sets["forest"]], key=len)
    out["N"] = len(best)
    out["bits"] = math.log2(len(best))
    gw = lambda S: ref - len(S)
    gb = lambda S: math.log2(ref / len(S)) if has_bound else float("nan")
    out["gap_words"] = gw(best)
    out["gap_bits"] = gb(best)
    for k, S in sets.items():
        out[f"N_{k}"] = len(S)
        out[f"gap_{k}_words"] = gw(S)
        out[f"gap_{k}_bits"] = gb(S)
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
    base = G.mask(base_set) if len(base_set) else np.zeros(G.n, np.bool_)
    eu, ev, starts = _component_edges(G)
    out = []
    for ci, comp in enumerate(comps):
        here = comp[base[comp]]
        too_large = (cfg_solver.get("ilp_max_edges") is not None
                     and starts[ci + 1] - starts[ci] > cfg_solver["ilp_max_edges"])
        if len(comp) <= 1 or too_large:
            # too large for the solver: keep the phase-1 set, which already
            # carries the greedy frequency tie-break and (1,1) frequency swaps
            out += here.tolist()
            continue
        r = exact_mis(G, comp, eu[starts[ci]:starts[ci + 1]], ev[starts[ci]:starts[ci + 1]],
                      cfg_solver["component_time_limit_s"], cfg_solver["workers"], hint=here,
                      weights=weights, fix_size=len(here), seed=seed)
        out += r["set"] if r["set"] and len(r["set"]) == len(here) else here.tolist()
    assert G.is_independent(out)
    return sorted(int(v) for v in out)


def caro_wei(G: Graph) -> float:
    return float(np.sum(1.0 / (G.degree() + 1.0)))


def knee(taus, N):
    """Kneedle (Satopää et al. 2011) on pool size N(tau), decreasing.  (bits vs
    tau is ~linear, so it has no knee; N is used instead.)

    N(tau) is S-shaped: flat at small tau (few edges), steepest drop, then a
    convex decreasing tail.  Kneedle is run as convex/decreasing on the tail,
    i.e. from the tau where N drops the most onward.
    The earlier rule (curve shape from the mean second difference) was a bug:
    that mean telescopes to (last step - first step), so the flat start alone
    labelled the Manhattan curve concave and Kneedle returned the grid edge."""
    from kneed import KneeLocator
    taus, N = np.asarray(taus, float), np.asarray(N, float)
    if len(taus) < 4:
        return None, None
    start = int(np.argmin(np.diff(N)))          # step with the steepest drop
    t, n = taus[start:], N[start:]
    if len(t) < 4:
        return None, None
    kl = KneeLocator(t, n, curve="convex", direction="decreasing")
    return (float(kl.knee) if kl.knee is not None else None), f"convex from tau={t[0]:g}"
