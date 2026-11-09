"""题目一：迷宫寻路（A* + 曼哈顿距离）

输入：
    start : Tuple[int, int]           起点坐标 (row, col)
    goal  : Tuple[int, int]           终点坐标 (row, col)
    grid  : List[List[str]]           地图，'.'/'S'/'E' 可通行，'X' 为障碍
输出：
    List[Tuple[int, int]]             起点到终点的最短路径；无解返回 []
"""

from __future__ import annotations

from typing import List, Tuple

from astar import astar, manhattan

Coord = Tuple[int, int]


def solve_maze(
    start: Coord,
    goal: Coord,
    grid: List[List[str]],
    use_heuristic: bool = True,
    return_expanded: bool = False,
):
    n_rows, n_cols = len(grid), len(grid[0])

    def passable(c: Coord) -> bool:
        r, cc = c
        return 0 <= r < n_rows and 0 <= cc < n_cols and grid[r][cc] != "X"

    h = (lambda c, g: manhattan(c, g)) if use_heuristic else (lambda c, g: 0.0)

    path, _, expanded = astar(
        start, goal, n_rows, n_cols, passable, step_cost=lambda a, b: 1, heuristic=h
    )
    result = path if path is not None else []

    if return_expanded:
        return result, expanded
    return result


if __name__ == "__main__":
    grid = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]
    print(solve_maze((0, 0), (3, 4), grid))
    # 期望: [(0,0), (0,1), (0,2), (1,2), (1,3), (1,4), (2,4), (3,4)]
    print(solve_maze((0, 0), (4, 0), grid))  # 期望: []
