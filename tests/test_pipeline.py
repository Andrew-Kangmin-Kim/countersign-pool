import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cpool import calibration, graph, phonetics
from cpool.hangul import decompose
from cpool.phonetics import Encoded, Params

P0 = Params([1] * 4, [1] * 4, 3.0, 6.0)
WORDS = "사과 사고 자고 학교 하고 가방 바다 나무 나물 나비 감자 호랑이 고양이".split()


def d(a, b, P=P0):
    E = Encoded([a, b])
    return float(phonetics.cross_distances(E.subset([0]), E.subset([1]), P)[0, 0])


def test_decompose():
    assert decompose("한") == ("ㅎ", "ㅏ", "ㄴ") and decompose("a") is None


def test_notebook_sanity():           # the notebook's own self-test
    assert d("불", "뿔") == pytest.approx(1.0)
    assert d("불", "뿔") < d("불", "학교")
    assert d("감자", "자감") > 0.5 and d("사과", "사과") == 0


def test_metric_and_blocking():
    E = Encoded(WORDS)
    rng = np.random.RandomState(0)
    for _ in range(4):
        P = Params(rng.rand(4) * 2, rng.rand(4) * 2, 3.0, 6.0)
        D = phonetics.cross_distances(E, E, P)
        assert np.allclose(D, D.T, atol=1e-5)
        for i, j, k in itertools.permutations(range(len(WORDS)), 3):
            assert D[i, k] <= D[i, j] + D[j, k] + 1e-4
        for i, j in itertools.combinations(range(len(WORDS)), 2):
            assert D[i, j] >= abs(int(E.LEN[i]) - int(E.LEN[j])) * 3.0 - 1e-5


def test_edge_list_equals_dense():
    E = Encoded(WORDS)
    D = phonetics.cross_distances(E, E, P0)
    EI, EJ, ED, _ = phonetics.edge_list(E, P0, 6.0, verbose=False)
    got = {(int(i), int(j)): float(x) for i, j, x in zip(EI, EJ, ED)}
    want = {(i, j): float(D[i, j]) for i, j in itertools.combinations(range(len(WORDS)), 2)
            if D[i, j] < 6.0}
    assert got.keys() == want.keys()


def test_feature_diff_reproduces_distance():
    P = Params([.4, .3, .5, 2.4], [1.1, 1.2, 1.8, .2], 3.0, 6.0)
    th = np.concatenate([P.cons_w, P.vowel_w, [3.0, 6.0]])
    for a, b in itertools.combinations(WORDS[:8], 2):
        assert phonetics.feature_diff(a, b, P) @ th == pytest.approx(d(a, b, P), abs=1e-3)


def test_sign_fix():
    coef = pd.Series({"c_place": -0.8, "c_manner": 0.5, "v_round": -0.02})
    cw, vw, signs = calibration._weights_from(coef, 0.1)
    assert cw[0] == pytest.approx(0.8)        # negative coef -> weight 0.8
    assert cw[1] == pytest.approx(0.1)        # wrong sign -> floor (notebook gave 0.5)
    assert vw[2] == pytest.approx(0.1)        # tiny -> floor
    assert signs["c_manner"] == "wrong_sign_floored"


def _rand_graph(n, p, seed):
    rng = np.random.RandomState(seed)
    e = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.rand() < p]
    return graph.Graph(n, np.array([a for a, _ in e], int), np.array([b for _, b in e], int))


def _alpha(G):
    for r in range(G.n, 0, -1):
        for S in itertools.combinations(range(G.n), r):
            if G.is_independent(S):
                return r


SV = {"ilp_time_limit_s": 10, "component_time_limit_s": 10, "workers": 2,
      "deterministic": True, "forest_runs": 20, "forest_k": 3, "swap_rounds": 6}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cell_exact_matches_bruteforce(seed):
    G = _rand_graph(14, 0.3, seed)
    r = graph.solve_cell(G, SV)
    a = _alpha(G)
    assert r["alpha_exact"] and r["reference"] == a == r["N"]
    for k in ("greedy_noswap", "greedy", "forest"):
        assert r[f"N_{k}"] <= a and r[f"gap_{k}_words"] == a - r[f"N_{k}"]


def test_forest_valid_and_reproducible():
    G = _rand_graph(300, 0.02, 7)
    act = G.degree() > 0
    g = graph.freq_swap(G, graph.two_improve(G, graph.mindeg_greedy(G)))
    f = graph.forest(G, runs=30, k=3, swap_rounds=6, seed=0)
    for S in (g, f):
        S = np.union1d(S, np.flatnonzero(~act))
        assert G.is_independent(S) and G.is_maximal(S)
    assert np.array_equal(f, graph.forest(G, runs=30, k=3, swap_rounds=6, seed=0))
    # k=1 randomized greedy is plain min-degree greedy
    assert np.array_equal(graph.randomized_greedy(G, k=1), graph.mindeg_greedy(G))


def test_greedy_matches_set_reference():
    """numba min-degree greedy == the original Python-set implementation"""
    from tests import _graph_sets_ref as ref
    for seed in range(5):
        G = _rand_graph(200, 0.04, seed)
        adj = [set(G.neighbors(v).tolist()) for v in range(G.n)]
        nodes = np.flatnonzero(G.degree() > 0).tolist()
        assert sorted(ref.mindeg_greedy(nodes, adj)) == graph.mindeg_greedy(G).tolist()


def test_forest_per_component_best():
    # two disjoint components: forest must keep the best run for EACH one
    G1 = _rand_graph(60, 0.08, 3)
    G = graph.Graph(120, np.concatenate([G1.ei, G1.ei + 60]), np.concatenate([G1.ej, G1.ej + 60]))
    lab = G.labels()
    runs = []
    for r in range(40):
        S = graph.freq_swap(G, graph.two_improve(G, graph.randomized_greedy(G, k=3, seed=1 * 100003 + r)))
        runs.append(S)
    f = graph.forest(G, runs=40, k=3, swap_rounds=6, seed=1)
    for c in np.unique(lab[G.degree() > 0]):
        best = max(np.sum(lab[S] == c) for S in runs)
        assert np.sum(lab[f] == c) == best      # best of all runs, per component
    assert G.is_independent(f) and G.is_maximal(np.union1d(f, np.flatnonzero(G.degree() == 0)))


def test_lexicographic_prefers_frequent():
    G = graph.Graph(5, np.array([0, 1, 3]), np.array([1, 2, 4]))
    r = graph.solve_cell(G, SV)
    S = graph.lexicographic_pool(G, r["_best"], {"component_time_limit_s": 5, "workers": 1},
                                 {v: 10 - v for v in range(5)})
    assert set(S) == {0, 2, 3}


def test_vectors_layout_and_closest_bruteforce():
    from cpool import vectors
    v = vectors.encode_one("바다")                 # ㅂ = (0,0,0,0) must still be "present"
    assert v[4] == 1.0 and v[3 * 5 * 2 + 4] == 0.0  # onset of syll 1 present; syll 3 empty
    assert vectors.encode_one("아이")[4] == 0.0     # silent ㅇ onset is absent
    rng = np.random.RandomState(0)
    V = rng.randint(0, 3, size=(60, vectors.DIM)).astype(np.float32)
    V[:, 4] = 1                                     # no all-zero vectors (cosine)
    ok = rng.rand(60) > 0.2
    group = np.arange(60)
    group[5] = group[7]                             # an "identical surface" pair
    for m in vectors.METRICS:
        got = vectors.closest_pairs(V, m, 25, ok, group, chunk=7)
        want = sorted((vectors.pair_distance(V[i], V[j], m), i, j)
                      for i in range(60) for j in range(i + 1, 60)
                      if ok[i] and ok[j] and group[i] != group[j])[:25]
        assert [(i, j) for i, j, _ in got] == [(i, j) for _, i, j in want], m
        assert np.allclose([d for *_, d in got], [d for d, *_ in want], atol=1e-5)
        allp = sorted(vectors.pair_distance(V[i], V[j], m) for i in range(60) for j in range(i + 1, 60)
                      if ok[i] and ok[j] and group[i] != group[j])
        tie = vectors.closest_pairs(V, m, 25, ok, group, chunk=7, include_ties=True)
        assert len(tie) == sum(x <= allp[24] + 1e-6 for x in allp), m
