"""统一测试入口：运行全部四题的官方样例并校验结果。

运行: python run_all_tests.py
"""

from __future__ import annotations

from problem1_maze import solve_maze
from problem2_weighted import solve_weighted
from problem3_alphabeta import build_sample_tree, solve_tree
from problem4_game import solve_game


def check(name, got, expected):
    ok = got == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got} expected={expected}")
    return ok


def main():
    results = []

    # ---------------- 题目一 ----------------
    grid1 = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]
    path = solve_maze((0, 0), (3, 4), grid1)
    results.append(check("Q1 迷宫路径", path,
                         [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]))
    results.append(check("Q1 无解返回[]", solve_maze((0, 0), (4, 0), grid1), []))

    # ---------------- 题目二 ----------------
    grid2 = [
        ["S", 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, "E"],
    ]
    results.append(check("Q2 最小代价", solve_weighted((0, 0), (4, 4), grid2), 7))

    # ---------------- 题目三 ----------------
    root = build_sample_tree()
    results.append(check("Q3 Minimax", solve_tree(root, use_alpha_beta=False), 5))
    results.append(check("Q3 Alpha-Beta", solve_tree(root, use_alpha_beta=True), 5))

    # ---------------- 题目四 ----------------
    res2, _ = solve_game(2, use_alpha_beta=True)
    results.append(check("Q4 H=2", res2, {"best_target": "T2", "best_value": -1}))
    res4, _ = solve_game(4, use_alpha_beta=True)
    results.append(check("Q4 H=4", res4, {"best_target": "T3", "best_value": 3}))

    print("\n" + ("[OK] 全部通过" if all(results) else "[FAIL] 存在失败用例"))


if __name__ == "__main__":
    main()
