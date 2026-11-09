"""题目二：加权地形（A*）

规则：
    - 数字代表进入该格的地形代价（进入 B 的代价 = cost(B)）
    - 'S' 起点格子代价视为 0；'E' 终点格子代价视为 0（题目规定最后一步没有代价）
    - 'X' 障碍不可通行
输出：
    int，最小总代价；无解返回 -1

关于启发函数可采纳性：
    路径上每进入一格至少花 min_cost（本题地形最小权值 = 1），
    因此 h(n) = manhattan(n, goal) * min_cost 不高估真实剩余代价，
    是"缩放后的曼哈顿距离"，仍然可采纳；用 *1 的普通曼哈顿距离也可采纳。
    若错误地乘上平均代价或最大代价，则会高估，破坏 A* 的最优性。
"""

from __future__ import annotations

from typing import List, Tuple

from astar import astar, manhattan

Coord = Tuple[int, int]


def _parse_terrain(cell) -> int | None:
    """'S' -> 0, 'E' -> 0, 'X' -> None(不可通行), 数字/数字字符串 -> int"""
    if cell in ("S", "E"):
        return 0
    if cell == "X":
        return None
    return int(cell)


def solve_weighted(start: Coord, goal: Coord, grid: List[List[str]]) -> int:
    n_rows, n_cols = len(grid), len(grid[0])
    terrain = [[_parse_terrain(cell) for cell in row] for row in grid]

    # 最小正权值，用于缩放启发函数保证可采纳
    min_cost = min(
        w for row in terrain for w in row if w is not None and w > 0
    )

    def passable(c: Coord) -> bool:
        r, cc = c
        return 0 <= r < n_rows and 0 <= cc < n_cols and terrain[r][cc] is not None

    def step_cost(a: Coord, b: Coord) -> int:
        return terrain[b[0]][b[1]]  # 进入 b 的代价

    def h(c: Coord, g: Coord) -> float:
        return manhattan(c, g) * min_cost

    path, cost, _ = astar(start, goal, n_rows, n_cols, passable, step_cost, h)
    return -1 if path is None else int(cost)


def solve_weighted_with_path(start: Coord, goal: Coord, grid: List[List[str]]):
    """调试用：同时返回路径，方便报告里展示代价累计过程。"""
    n_rows, n_cols = len(grid), len(grid[0])
    terrain = [[_parse_terrain(cell) for cell in row] for row in grid]
    min_cost = min(w for row in terrain for w in row if w is not None and w > 0)

    def passable(c: Coord) -> bool:
        r, cc = c
        return 0 <= r < n_rows and 0 <= cc < n_cols and terrain[r][cc] is not None

    path, cost, expanded = astar(
        start, goal, n_rows, n_cols, passable,
        step_cost=lambda a, b: terrain[b[0]][b[1]],
        heuristic=lambda c, g: manhattan(c, g) * min_cost,
    )
    # 打印逐步累计
    if path:
        acc, printout = 0, []
        for i, p in enumerate(path):
            if i > 0:
                acc += terrain[p[0]][p[1]]
            printout.append(f"{p}(+{terrain[p[0]][p[1]] if i else 0} => {acc})")
        print(" -> ".join(printout))
    return (path, int(cost)) if path else (None, -1)


if __name__ == "__main__":
    grid = [
        ["S", 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, "E"],
    ]
    print(solve_weighted((0, 0), (4, 4), grid))  # 期望: 7
    solve_weighted_with_path((0, 0), (4, 4), grid)
