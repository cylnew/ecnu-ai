"""题目四：竞争环境下的宝藏分配（A* + Alpha-Beta 综合题）

地图（7x7）：X 障碍；数字为进入代价；A、B、T 所在格代价为 1。
    0 1 2 3 4 5 6
0   A 1 1 X 2 1 T1
1   1 X 6 X 1 X 1
2   1 1 T2 1 1 1 1
3   X 1 X 8 X 1 X
4   1 1 1 1 T3 1 1
5   1 X 9 X 1 X 1
6   T4 1 1 X 1 1 B

规则：
    - 行动顺序 A -> B -> A -> B，共 4 回合，每回合选一个剩余宝藏
    - 移动代价用 A* 计算（进入格子计入该格代价）
    - 收益：Gain(p, Ti) = Vi - d(p, Ti)，U = Score_A - Score_B
    - A 最大化 U，B 最小化 U
任务：
    1. 输出 A、B、T1..T4 两两之间的最小移动代价 6x6 距离矩阵
    2. 设计博弈状态
    3. H=2、H=4 时分别用 Minimax 与 Alpha-Beta 求 A 的最优首步与最优 U
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

from astar import astar, manhattan

Coord = Tuple[int, int]

RAW_MAP = [
    [1, 1, 1, None, 2, 1, 1],   # T1 (0,6) 格代价 1
    [1, None, 6, None, 1, None, 1],
    [1, 1, 1, 1, 1, 1, 1],      # T2 (2,2)
    [None, 1, None, 8, None, 1, None],
    [1, 1, 1, 1, 1, 1, 1],      # T3 (4,4)
    [1, None, 9, None, 1, None, 1],
    [1, 1, 1, None, 1, 1, 1],   # T4 (6,0)
]

A_POS: Coord = (0, 0)
B_POS: Coord = (6, 6)
TREASURES: Dict[str, Tuple[Coord, int]] = {
    "T1": ((0, 6), 9),
    "T2": ((2, 2), 6),
    "T3": ((4, 4), 7),
    "T4": ((6, 0), 7),
}

N_ROWS, N_COLS = len(RAW_MAP), len(RAW_MAP[0])

# 模块级距离矩阵缓存：solve_game 被多次调用（H=2/H=4 × Minimax/AB）时，
# 36 次 A* 只需真正跑一次（P1 优化：缓存）。
_DIST_CACHE: Optional[Dict[Tuple[str, str], int]] = None


def passable(c: Coord) -> bool:
    return (
        0 <= c[0] < N_ROWS
        and 0 <= c[1] < N_COLS
        and RAW_MAP[c[0]][c[1]] is not None
    )


def step_cost(a: Coord, b: Coord) -> int:
    return RAW_MAP[b[0]][b[1]]  # 进入 b 的代价


def min_cost_between(p: Coord, q: Coord) -> int:
    """用 A* 求 p 到 q 的最小移动代价；终点格代价也计入（T/A/B 格为 1）。"""
    path, cost, _ = astar(
        p, q, N_ROWS, N_COLS, passable, step_cost,
        heuristic=lambda c, g: manhattan(c, g) * 1,  # 最小权值为 1
    )
    return -1 if path is None else int(cost)


def build_distance_matrix(verbose: bool = True, use_cache: bool = True) -> Dict[Tuple[str, str], int]:
    """计算 A、B、T1..T4 两两之间的最小移动代价。

    use_cache=True 时复用模块级缓存（P1 优化：solve_game 被调用多次时，
    36 次 A* 只需真正跑一次）。
    """
    global _DIST_CACHE
    if use_cache and _DIST_CACHE is not None:
        if verbose:
            _print_matrix(_DIST_CACHE)
        return _DIST_CACHE

    entities: Dict[str, Coord] = {"A": A_POS, "B": B_POS}
    entities.update({name: pos for name, (pos, _) in TREASURES.items()})
    names = ["A", "B"] + list(TREASURES.keys())

    dist: Dict[Tuple[str, str], int] = {}
    for u, v in itertools.product(names, names):
        dist[(u, v)] = 0 if u == v else min_cost_between(entities[u], entities[v])

    if use_cache:
        _DIST_CACHE = dist
    if verbose:
        _print_matrix(dist)
    return dist


def _print_matrix(dist: Dict[Tuple[str, str], int]) -> None:
    names = ["A", "B"] + list(TREASURES.keys())
    print("距离矩阵（行/列: A B T1 T2 T3 T4）")
    print("        " + "".join(f"{n:>6}" for n in names))
    for u in names:
        print(f"{u:>4}    " + "".join(f"{dist[(u, v)]:>6}" for v in names))


# ------------------------------------------------------------------
# 博弈搜索
# ------------------------------------------------------------------
class GameState:
    """博弈状态：双方位置、剩余宝藏、累计收益、当前玩家。"""

    __slots__ = ("pos_a", "pos_b", "remaining", "score_a", "score_b", "is_max")

    def __init__(self, pos_a, pos_b, remaining, score_a=0, score_b=0, is_max=True):
        self.pos_a = pos_a
        self.pos_b = pos_b
        self.remaining = remaining
        self.score_a = score_a
        self.score_b = score_b
        self.is_max = is_max

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b

    def successors(self, dist, treasures, sort_moves: bool = False):
        """生成 选择每个剩余宝藏 后的新状态。

        sort_moves=True 时按即时 gain 降序排列后继（Move Ordering 优化：
        MAX 与 MIN 层都优先尝试即时收益最大的宝藏，使 alpha/beta 窗口
        尽早收紧，触发更多剪枝；不影响返回的最优值）。
        """
        moves = []
        for t in self.remaining:
            if self.is_max:
                d = dist[(pos_key(self.pos_a), t)]
                gain = treasures[t][1] - d
                rest = tuple(x for x in self.remaining if x != t)
                # 移动后玩家的位置更新为该宝藏所在格
                moves.append((gain, t, GameState(treasures[t][0], self.pos_b, rest,
                                   self.score_a + gain, self.score_b, False)))
            else:
                d = dist[(pos_key(self.pos_b), t)]
                gain = treasures[t][1] - d
                rest = tuple(x for x in self.remaining if x != t)
                moves.append((gain, t, GameState(self.pos_a, treasures[t][0], rest,
                                   self.score_a, self.score_b + gain, True)))
        if sort_moves:
            # MAX/MIN 都优先试 gain 最大的：MAX 直接增大 U，
            # MIN 增大 score_b 从而更快压低 U，两者都能更快收紧窗口
            moves.sort(key=lambda m: m[0], reverse=True)
        for _, t, ns in moves:
            yield t, ns


def pos_key(pos: Coord) -> str:
    """把坐标映射回实体名（A/B/Tx），用于查距离矩阵。"""
    if pos == A_POS:
        return "A"
    if pos == B_POS:
        return "B"
    for name, (p, _) in TREASURES.items():
        if p == pos:
            return name
    raise ValueError(f"未知位置: {pos}")


def minimax(state: GameState, depth: int, dist, treasures, stats=None, sort_moves: bool = False) -> int:
    if stats is None:
        stats = {}
    stats["visited"] = stats.get("visited", 0) + 1

    if depth == 0 or not state.remaining:
        return state.utility

    values = [
        minimax(ns, depth - 1, dist, treasures, stats, sort_moves)
        for _, ns in state.successors(dist, treasures, sort_moves)
    ]
    return max(values) if state.is_max else min(values)


def alpha_beta(state, depth, dist, treasures, alpha=-float("inf"), beta=float("inf"), stats=None, sort_moves: bool = False):
    if stats is None:
        stats = {}
    stats["visited"] = stats.get("visited", 0) + 1

    if depth == 0 or not state.remaining:
        return state.utility

    if state.is_max:
        value = -float("inf")
        for _, ns in state.successors(dist, treasures, sort_moves):
            value = max(value, alpha_beta(ns, depth - 1, dist, treasures, alpha, beta, stats, sort_moves))
            alpha = max(alpha, value)
            if value >= beta:
                stats["pruned"] = stats.get("pruned", 0) + 1
                break
        return value
    else:
        value = float("inf")
        for _, ns in state.successors(dist, treasures, sort_moves):
            value = min(value, alpha_beta(ns, depth - 1, dist, treasures, alpha, beta, stats, sort_moves))
            beta = min(beta, value)
            if value <= alpha:
                stats["pruned"] = stats.get("pruned", 0) + 1
                break
        return value


def solve_game(depth: int, use_alpha_beta: bool = True, sort_moves: bool = False):
    dist = build_distance_matrix(verbose=False)
    root = GameState(A_POS, B_POS, tuple(TREASURES.keys()), 0, 0, True)
    stats = {"visited": 0, "pruned": 0}

    best_target, best_value = None, -float("inf")
    for t, child in root.successors(dist, TREASURES, sort_moves):
        v = (
            alpha_beta(child, depth - 1, dist, TREASURES, -float("inf"), float("inf"), stats, sort_moves)
            if use_alpha_beta
            else minimax(child, depth - 1, dist, TREASURES, stats, sort_moves)
        )
        if v > best_value:
            best_target, best_value = t, v
    return {"best_target": best_target, "best_value": int(best_value)}, stats


if __name__ == "__main__":
    build_distance_matrix()
    for h in (2, 4):
        for name, use_ab in (("Minimax", False), ("Alpha-Beta", True)):
            res, stats = solve_game(h, use_ab)
            print(f"H={h} {name:<10} -> {res}   stats={stats}")
    # 期望: H=2 => {'best_target': 'T2', 'best_value': -1}
    #       H=4 => {'best_target': 'T3', 'best_value': 3}
