"""A* 通用实现（题目一、题目二、题目四共用）。

核心思想：
    维护两个表
        g(n): 从起点到 n 的实际最小代价
        f(n) = g(n) + h(n): 估计的总代价
    每次从优先队列中弹出 f 最小的节点扩展，直到弹出终点。
    只要 h 可采纳（不高估真实剩余代价），第一次弹出终点时的 g 即为最优解。
"""

from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Optional, Tuple

Coord = Tuple[int, int]

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上 下 左 右


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(
    start: Coord,
    goal: Coord,
    n_rows: int,
    n_cols: int,
    passable: Callable[[Coord], bool],
    step_cost: Callable[[Coord, Coord], float] = lambda a, b: 1.0,
    heuristic: Optional[Callable[[Coord, Coord], float]] = None,
    weight: float = 1.0,
) -> Tuple[Optional[List[Coord]], float, int]:
    """在 (n_rows x n_cols) 的网格上做 A* 搜索。

    参数
    ----
    passable(c)      : 格子 c 是否可通行
    step_cost(a, b)  : 从 a 走到相邻格 b 的代价
    heuristic(c, g)  : 启发函数 h(c)，默认为 0（退化成 Dijkstra）
    weight           : 加权 A* 系数 w，f = g + w*h；w=1 为标准 A*，
                       w>1 牺牲最优性换取更少扩展（P1 优化实验用）

    返回
    ----
    (path, cost, expanded)
        path     : 从起点到终点的坐标列表（含两端）；无解为 None
        cost     : 路径总代价；无解为 inf
        expanded : 弹出并扩展的节点数（用于实验对照分析）

    注：堆中存 (f, h, 序号, 节点)，f 相同时优先扩展 h 更小的节点
    （Tie-breaking，偏向离终点更近的分支，可减少扩展）。
    """
    heuristic = heuristic or (lambda c, g: 0.0)

    if not passable(start) or not passable(goal):
        return None, float("inf"), 0

    g_score: Dict[Coord, float] = {start: 0.0}
    parent: Dict[Coord, Coord] = {}
    # 堆中存 (f, h, 序号, 节点)：
    #  f 相同时优先扩展 h 更小的节点（Tie-breaking，偏向离终点更近的分支）；
    #  序号用于在 f、h 都相同时避免比较坐标元组
    open_heap: List[Tuple[float, float, int, Coord]] = []
    counter = 0
    h0 = heuristic(start, goal)
    heapq.heappush(open_heap, (h0 * weight, h0, counter, start))
    closed = set()
    expanded = 0

    while open_heap:
        f, _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue  # 堆中的陈旧条目

        if cur == goal:
            # 终点第一次被弹出 => g 最优（h 可采纳）
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            return path, g_score[goal], expanded

        closed.add(cur)
        expanded += 1

        r, c = cur
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] < n_rows and 0 <= nxt[1] < n_cols):
                continue
            if nxt in closed or not passable(nxt):
                continue

            tentative_g = g_score[cur] + step_cost(cur, nxt)
            if tentative_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative_g
                parent[nxt] = cur
                counter += 1
                hn = heuristic(nxt, goal)
                heapq.heappush(
                    open_heap, (tentative_g + weight * hn, hn, counter, nxt)
                )

    return None, float("inf"), expanded
