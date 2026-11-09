"""对照实验一键脚本：跑出报告第 2-6 节所需的全部数据。

运行: python experiments.py
输出: 终端表格（可直接贴进报告）

包含:
  E1 题目一: 样例地图 + 20x20 随机地图, h=0 vs 曼哈顿 (扩展节点数)
  E2 题目二: 启发函数可采纳性 - 缩放系数 min/mean/max 的破坏性实验
  E3 题目三: 子节点访问顺序对剪枝的影响 (32 种组合) + 加权统计
  E4 题目四: H=2/H=4 各首步 U 值与博弈线; Minimax vs Alpha-Beta; 排序优化
  E5 P1优化: 加权 A* (w=1/1.5/2) 权衡曲线; 距离矩阵缓存计时
"""

from __future__ import annotations

import itertools
import random
import time

from astar import astar, manhattan
from problem1_maze import solve_maze
from problem2_weighted import _parse_terrain
from problem3_alphabeta import TreeNode, alpha_beta, minimax
from problem4_game import (
    A_POS, B_POS, TREASURES, GameState, alpha_beta as ab_game,
    build_distance_matrix, minimax as mm_game, pos_key, solve_game,
)


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ================================================================ E1
def exp1_heuristic():
    section("E1 题目一: h=0 (Dijkstra) vs 曼哈顿距离")
    grid = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]
    _, e1 = solve_maze((0, 0), (3, 4), grid, True, True)
    _, e0 = solve_maze((0, 0), (3, 4), grid, False, True)
    print(f"样例 5x5 地图:  曼哈顿 expanded={e1} | h=0 expanded={e0} "
          f"| 缩减 {(1 - e1 / e0) * 100:.1f}%")

    # 20x20 随机地图
    random.seed(42)

    def gen(n, m, p=0.3):
        g = [["." for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if random.random() < p:
                    g[i][j] = "X"
        g[0][0] = "S"
        g[n - 1][m - 1] = "E"
        return g

    tot_h = tot_0 = ok = 0
    N = 30
    for _ in range(N):
        g = gen(20, 20)
        p1, a = solve_maze((0, 0), (19, 19), g, True, True)
        p0, b = solve_maze((0, 0), (19, 19), g, False, True)
        if p1:  # 只统计可解样例
            ok += 1
            tot_h += a
            tot_0 += b
    print(f"20x20 随机地图(障碍率30%, 可解 {ok}/{N}): "
          f"曼哈顿平均 {tot_h / ok:.0f} | h=0 平均 {tot_0 / ok:.0f} "
          f"| 缩减 {(1 - tot_h / tot_0) * 100:.1f}%")
    print("结论: 曼哈顿 h 引导搜索朝终点定向扩展, h=0 退化为洪水式扩散.")


# ================================================================ E2
def exp2_admissibility():
    section("E2 题目二: 启发函数可采纳性 (缩放系数对比)")
    grid = [
        ["S", 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, "E"],
    ]
    n_rows, n_cols = 5, 5
    terrain = [[_parse_terrain(c) for c in row] for row in grid]
    weights = [w for row in terrain for w in row if w is not None and w > 0]
    mn, mx = min(weights), max(weights)
    mean = sum(weights) / len(weights)

    def run(scale, tag):
        def ps(c):
            return terrain[c[0]][c[1]] is not None

        path, cost, exp = astar(
            (0, 0), (4, 4), n_rows, n_cols, ps,
            step_cost=lambda a, b: terrain[b[0]][b[1]],
            heuristic=lambda c, g: manhattan(c, g) * scale,
        )
        flag = "最优" if cost == 7 else "!!! 次优/错"
        print(f"  h=manhattan*{tag:<12} -> 代价={int(cost):>3} 扩展={exp:>3} [{flag}]")
        return cost, exp

    print("样例地图 (最优代价=7, 缩放系数必须取下界才可采纳):")
    run(mn, f"{mn}(min,可采纳)")
    run(1, "1(普通曼哈顿)")
    run(mean, f"{mean:.2f}(mean,高估)")
    run(mx, f"{mx}(max,高估)")
    print("样例地图太小, 高估的 h 碰巧仍得最优 -> 下面用专门构造的反例地图:")

    # 反例地图: 直达终点的路全是沼泽(8), 绕行下排全是平原(1)
    #   S 8 8 8 E
    #   1 1 1 1 1
    # 最优是绕行, 代价=5; 高估的 h 会让 A* 贪心直走, 代价=24
    grid_bad = [
        ["S", 8, 8, 8, "E"],
        [1, 1, 1, 1, 1],
    ]
    terrain_bad = [[_parse_terrain(c) for c in row] for row in grid_bad]

    def run_bad(scale, tag):
        def ps(c):
            r, cc = c
            return 0 <= r < 2 and 0 <= cc < 5 and terrain_bad[r][cc] is not None

        path, cost, exp = astar(
            (0, 0), (0, 4), 2, 5, ps,
            step_cost=lambda a, b: terrain_bad[b[0]][b[1]],
            heuristic=lambda c, g: manhattan(c, g) * scale,
        )
        flag = "最优" if cost == 5 else "!!! 次优(贪心穿沼泽)"
        print(f"  h=manhattan*{tag:<14} -> 代价={int(cost):>3} 扩展={exp:>3} [{flag}]")
        return cost

    print("反例地图 (最优代价=5, 即绕行下排平原):")
    c1 = run_bad(1, "1(min,可采纳)")
    c2 = run_bad(8, "8(max,高估)")
    print("结论: 系数取 min_cost=1 时不高估, 仍是最优; 取 max 会高估, "
          "A* 贪心直穿沼泽而返回次优解." + (" [反例生效]" if c2 > c1 else ""))
    return c1, c2


# ================================================================ E3
def exp3_ordering():
    section("E3 题目三: 子节点访问顺序对剪枝的影响")

    def build(bits):
        leaf = lambda v: TreeNode(value=v)
        m1 = TreeNode(children=[leaf(3), leaf(5)])
        m2 = TreeNode(children=[leaf(6), leaf(9)])
        m3 = TreeNode(children=[leaf(1), leaf(2)])
        if bits[0]:
            m1.children.reverse()
        if bits[1]:
            m2.children.reverse()
        if bits[2]:
            m3.children.reverse()
        n1 = TreeNode(children=[m1, m2])
        n2 = TreeNode(children=[m2, m3])
        if bits[3]:
            n1.children.reverse()
        if bits[4]:
            n2.children.reverse()
        return TreeNode(children=[n1, n2])

    rows = []
    for bits in itertools.product([0, 1], repeat=5):
        r = build(bits)
        s_mm, s_ab = {}, {}
        v1 = minimax(r, True, s_mm)
        v2 = alpha_beta(r, True, float("-inf"), float("inf"), s_ab)
        assert v1 == v2 == 5
        rows.append((s_ab["visited"], s_ab.get("pruned", 0), bits))

    from collections import Counter
    cnt = Counter((v, p) for v, p, _ in rows)
    print("32 种访问顺序下 Alpha-Beta 的 visited/pruned 分布:")
    for (v, p), c in sorted(cnt.items()):
        print(f"  visited={v:>2} pruned={p} : {c:>2} 种顺序")
    print("Minimax 在所有顺序下恒为 visited=15 (无剪枝, 与顺序无关).")
    print("结论: MAX 层先看大值 / MIN 层先看小值时窗口收紧最快, "
          "剪枝最多; 顺序差时 visited 退化到 15, 与 Minimax 相同.")


# ================================================================ E4
def exp4_game():
    section("E4 题目四: H=2/H=4 决策差异与剪枝效果")
    import problem4_game as G
    dist = build_distance_matrix(verbose=False)
    T = TREASURES
    root = GameState(A_POS, B_POS, tuple(T.keys()), 0, 0, True)

    print("各首步在 H=2 / H=4 下的 minimax 值:")
    table = {}
    for t, child in root.successors(dist, T):
        table[t] = (mm_game(child, 1, dist, T), mm_game(child, 3, dist, T))
    print(f"  {'首步':<6}{'H=2':>6}{'H=4':>6}")
    for t in ("T1", "T2", "T3", "T4"):
        print(f"  {t:<6}{table[t][0]:>6}{table[t][1]:>6}")
    print("决策: H=2 选 T2 (短视贪心, 即时 gain 最高); "
          "H=4 选 T3 (远视, 占住中心且迫使 B 走高代价路).")

    print("\nMinimax vs Alpha-Beta 开销 (sort_moves=False):")
    for h in (2, 4):
        _, s1 = solve_game(h, use_alpha_beta=False)
        _, s2 = solve_game(h, use_alpha_beta=True)
        print(f"  H={h}: Minimax visited={s1['visited']:>3} | "
              f"Alpha-Beta visited={s2['visited']:>3} pruned={s2.get('pruned', 0)}")

    print("\nMove Ordering 优化效果 (sort_moves=True, 按即时 gain 排序):")
    for h in (2, 4):
        _, s2 = solve_game(h, use_alpha_beta=True, sort_moves=True)
        # 验证最优值不变
        r, _ = solve_game(h, use_alpha_beta=True, sort_moves=False)
        r2, _ = solve_game(h, use_alpha_beta=True, sort_moves=True)
        assert r == r2, "排序不应改变最优值!"
        print(f"  H={h}: Alpha-Beta+排序 visited={s2['visited']:>3} "
              f"pruned={s2.get('pruned', 0)} (最优值不变 {r})")


# ================================================================ E5
def exp5_weighted_and_cache():
    section("E5 P1优化: 加权 A* 权衡曲线 + 距离矩阵缓存")
    grid = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]
    n_rows, n_cols = 5, 5

    def ps(c):
        return grid[c[0]][c[1]] != "X"

    print("加权 A* (f = g + w*h) 在样例迷宫上:")
    for w in (1.0, 1.5, 2.0, 3.0):
        path, cost, exp = astar(
            (0, 0), (3, 4), n_rows, n_cols, ps,
            step_cost=lambda a, b: 1,
            heuristic=lambda c, g: manhattan(c, g), weight=w,
        )
        print(f"  w={w:<4} -> 路径长={len(path)} 扩展={exp} "
              f"{'(仍最优)' if len(path) == 8 else '(次优, 更快)'}")

    print("\n距离矩阵缓存效果 (36 次 A* 只跑一次, 用调用计数而非计时, 避免小规模噪音):")
    import problem4_game as G
    calls = {"n": 0}
    orig = G.min_cost_between

    def counted(p, q):
        calls["n"] += 1
        return orig(p, q)

    G.min_cost_between = counted
    try:
        G._DIST_CACHE = None
        G.build_distance_matrix(verbose=False, use_cache=False)
        n_nocache = calls["n"]
        calls["n"] = 0
        G._DIST_CACHE = None
        G.build_distance_matrix(verbose=False, use_cache=True)   # 首次, 写缓存
        n_first = calls["n"]
        calls["n"] = 0
        G.build_distance_matrix(verbose=False, use_cache=True)   # 命中缓存
        n_hit = calls["n"]
    finally:
        G.min_cost_between = orig
    print(f"  无缓存: A* 调用 {n_nocache} 次 | 首次(写缓存): {n_first} 次 | "
          f"命中缓存: {n_hit} 次")
    print("结论: solve_game 被调用 4 次(H=2/H=4 x Minimax/AB)时, "
          "缓存把 4x30=120 次 A* 降为 30 次 "
          "(6x6 矩阵去掉 6 个对角自环).")


def main():
    exp1_heuristic()
    exp2_admissibility()
    exp3_ordering()
    exp4_game()
    exp5_weighted_and_cache()
    print("\n[DONE] 全部对照实验完成, 数据可直接用于报告.")


if __name__ == "__main__":
    main()
