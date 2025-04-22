#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import random
from copy import deepcopy
import math
import time  # 新增：用于计时
import numpy as np
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model


########################################
# 1) 计算行/列提示的辅助函数
########################################
def compute_row_col_hints(grid):
    m = len(grid)
    n = len(grid[0]) if m else 0
    row_hints = []
    for i in range(m):
        count = 0
        hints = []
        for val in grid[i]:
            if val == 1:
                count += 1
            else:
                if count > 0:
                    hints.append(count)
                    count = 0
        if count > 0:
            hints.append(count)
        row_hints.append(hints)

    col_hints = []
    for j in range(n):
        count = 0
        hints = []
        for i in range(m):
            if grid[i][j] == 1:
                count += 1
            else:
                if count > 0:
                    hints.append(count)
                    count = 0
        if count > 0:
            hints.append(count)
        col_hints.append(hints)
    return row_hints, col_hints


########################################
# 2) 建立 Nonogram 模型（CP-SAT）
########################################
def build_multi_block_cp_model(m, n, row_constraints, col_constraints):
    model = cp_model.CpModel()
    x = []
    for i in range(m):
        row_vars = []
        for j in range(n):
            var = model.NewBoolVar(f"x_{i}_{j}")
            row_vars.append(var)
        x.append(row_vars)
    # 行约束
    for i in range(m):
        blocks = row_constraints[i]
        if not blocks:
            for j in range(n):
                model.Add(x[i][j] == 0)
            continue
        rowBlockVars = []
        for b, length_b in enumerate(blocks):
            starts = []
            for start_col in range(n - length_b + 1):
                starts.append(model.NewBoolVar(f"rB_{i}_{b}_{start_col}"))
            model.Add(sum(starts) == 1)
            rowBlockVars.append((length_b, starts))

        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            starts_b = rowBlockVars[b][1]
            starts_next = rowBlockVars[b + 1][1]
            for s1, var1 in enumerate(starts_b):
                for s2, var2 in enumerate(starts_next):
                    if s2 < s1 + length_b + 1:
                        model.Add(var1 + var2 <= 1)

        for j in range(n):
            cover_vars = []
            for (length_b, starts) in rowBlockVars:
                for s, var_s in enumerate(starts):
                    if s <= j < s + length_b:
                        cover_vars.append(var_s)
            if cover_vars:
                model.Add(x[i][j] == sum(cover_vars))
            else:
                model.Add(x[i][j] == 0)

    # 列约束
    for j in range(n):
        blocks = col_constraints[j]
        if not blocks:
            for i in range(m):
                model.Add(x[i][j] == 0)
            continue
        colBlockVars = []
        for b, length_b in enumerate(blocks):
            starts = []
            for start_row in range(m - length_b + 1):
                starts.append(model.NewBoolVar(f"cB_{j}_{b}_{start_row}"))
            model.Add(sum(starts) == 1)
            colBlockVars.append((length_b, starts))

        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            starts_b = colBlockVars[b][1]
            starts_next = colBlockVars[b + 1][1]
            for s1, var1 in enumerate(starts_b):
                for s2, var2 in enumerate(starts_next):
                    if s2 < s1 + length_b + 1:
                        model.Add(var1 + var2 <= 1)

        for i in range(m):
            cover_vars = []
            for (length_b, starts) in colBlockVars:
                for s, var_s in enumerate(starts):
                    if s <= i < s + length_b:
                        cover_vars.append(var_s)
            if cover_vars:
                model.Add(x[i][j] == sum(cover_vars))
            else:
                model.Add(x[i][j] == 0)

    return model, x


########################################
# 3) 求解器回调：收集所有解
########################################
class NonogramAllSolutionsCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, x, max_solutions=1000):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._x = x
        self._solutions = []
        self._m = len(x)
        self._n = len(x[0]) if self._m > 0 else 0
        self._max_solutions = max_solutions

    def OnSolutionCallback(self):
        sol_matrix = []
        for i in range(self._m):
            row_vals = []
            for j in range(self._n):
                row_vals.append(self.Value(self._x[i][j]))
            sol_matrix.append(row_vals)
        self._solutions.append(sol_matrix)
        if len(self._solutions) >= self._max_solutions:
            self.StopSearch()

    def solutions(self):
        return self._solutions


########################################
# 4) 数解函数（调用 CP-SAT）
########################################
def count_solutions_up_to_1000(puzzle_data, time_limit=10.0):
    row_hints = puzzle_data.get("row_hints", puzzle_data.get("row_constraints"))
    col_hints = puzzle_data.get("col_hints", puzzle_data.get("col_constraints"))
    if not row_hints or not col_hints:
        return 0
    m = len(row_hints)
    n = len(col_hints)
    if m == 0 or n == 0:
        return 0

    model, x = build_multi_block_cp_model(m, n, row_hints, col_hints)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    collector = NonogramAllSolutionsCollector(x, max_solutions=1000)
    status = solver.SearchForAllSolutions(model, collector)

    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return 0
    return len(collector.solutions())


########################################
# 5) 计算歧义格出现频率
########################################
def compute_ambiguous_cells_with_frequency(solutions, original_grid):
    freq = {}
    if not solutions or original_grid is None:
        return freq
    m = len(original_grid)
    n = len(original_grid[0]) if m > 0 else 0
    for sol in solutions:
        for i in range(m):
            for j in range(n):
                if sol[i][j] != original_grid[i][j]:
                    freq[(i, j)] = freq.get((i, j), 0) + 1
    return freq


########################################
# 6) 连通性检测函数及孤立黑格检测
########################################
def count_black_components(grid):
    m = len(grid)
    if m == 0:
        return 0
    n = len(grid[0])
    visited = [[False] * n for _ in range(m)]

    def dfs(i, j):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= m or cj < 0 or cj >= n:
                continue
            if visited[ci][cj] or grid[ci][cj] == 0:
                continue
            visited[ci][cj] = True
            for di, dj in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1), (0, 1),
                           (1, -1), (1, 0), (1, 1)]:
                stack.append((ci + di, cj + dj))

    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1 and not visited[i][j]:
                count += 1
                dfs(i, j)
    return count


def count_white_components(grid):
    inverted = [[1 - cell for cell in row] for row in grid]
    return count_black_components(inverted)


def is_isolated_black(grid, i, j):
    m = len(grid)
    n = len(grid[0]) if m > 0 else 0
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < m and 0 <= nj < n:
            if grid[ni][nj] == 1:
                return False
    return True


########################################
# 7) 边界判断：判断单独翻转一个格子是否允许
########################################
def allowed_to_flip(original_grid, i, j):
    orig_black = count_black_components(original_grid)
    orig_white = count_white_components(original_grid)

    candidate = deepcopy(original_grid)
    candidate[i][j] = 1 - candidate[i][j]

    new_black = count_black_components(candidate)
    new_white = count_white_components(candidate)
    if new_black > orig_black or new_white > orig_white:
        return False
    return True


########################################
# 8) 加权贪心微调函数（增加总指标输出）
########################################
def greedy_refinement_weighted(puzzle_data, max_iterations=1000, time_limit=10.0):
    """
    基于加权频率的贪心微调：
      - 每次 CP-SAT 枚举部分解(最多 50 个) -> 计算与 current_state 不同的格子出现频率
      - 按出现频率加权随机选一个歧义格翻转(若不破坏连通性)
      - 若解数降低则接受
      - 直到唯一解或达到 max_iterations

    现在会返回包括收敛时间、迭代次数、最终汉明距离、success等信息
    """
    original_grid = deepcopy(puzzle_data["grid"])
    current_state = deepcopy(original_grid)

    # 预先补齐提示
    row_hints, col_hints = compute_row_col_hints(current_state)
    puzzle_data["row_hints"] = row_hints
    puzzle_data["col_hints"] = col_hints

    start_time = time.time()
    iteration = 0

    # 循环微调
    while iteration < max_iterations:
        iteration += 1

        # 计算当前解数
        candidate_puzzle = {
            "grid": current_state,
            "row_hints": compute_row_col_hints(current_state)[0],
            "col_hints": compute_row_col_hints(current_state)[1]
        }
        sol_count = count_solutions_up_to_1000(candidate_puzzle, time_limit=time_limit)
        if sol_count == 1:
            # 已经唯一解了，退出
            break

        # 使用 CP-SAT 枚举最多 50 个解, 找出所有歧义格及出现频率
        m_val = len(current_state)
        n_val = len(current_state[0])
        model, x = build_multi_block_cp_model(m_val, n_val,
                                              puzzle_data["row_hints"],
                                              puzzle_data["col_hints"])
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        collector = NonogramAllSolutionsCollector(x, max_solutions=50)
        solver.SearchForAllSolutions(model, collector)
        solutions = collector.solutions()
        ambiguous_freq = compute_ambiguous_cells_with_frequency(solutions, current_state)
        if not ambiguous_freq:
            # 无歧义格可改
            break

        # 采用频率加权随机选择一个格子翻转
        cells = list(ambiguous_freq.keys())
        weights = list(ambiguous_freq.values())
        i_sel, j_sel = random.choices(cells, weights=weights, k=1)[0]

        # 判断是否允许翻转(连通性不增加)
        if not allowed_to_flip(current_state, i_sel, j_sel):
            continue

        candidate_state = deepcopy(current_state)
        candidate_state[i_sel][j_sel] = 1 - candidate_state[i_sel][j_sel]

        # 计算翻转后解数
        new_puzzle = {
            "grid": candidate_state,
            "row_hints": compute_row_col_hints(candidate_state)[0],
            "col_hints": compute_row_col_hints(candidate_state)[1]
        }
        new_sol_count = count_solutions_up_to_1000(new_puzzle, time_limit=time_limit)

        # 若解数降低，则接受
        if new_sol_count < sol_count:
            current_state = candidate_state
            puzzle_data["row_hints"], puzzle_data["col_hints"] = compute_row_col_hints(current_state)

    # 循环结束后，统计结果
    end_time = time.time()
    final_sol_count = count_solutions_up_to_1000({
        "grid": current_state,
        "row_hints": compute_row_col_hints(current_state)[0],
        "col_hints": compute_row_col_hints(current_state)[1]
    }, time_limit=time_limit)

    # 计算最终汉明距离
    diff_count = 0
    for r1, r2 in zip(original_grid, current_state):
        for a, b in zip(r1, r2):
            if a != b:
                diff_count += 1

    success = (final_sol_count == 1)

    return {
        "final_grid": current_state,
        "iteration_used": iteration,
        "time_spent": end_time - start_time,
        "final_hamming_distance": diff_count,
        "success": success
    }


########################################
# 9) 对比图函数
########################################
def save_comparison_image(original_grid, refined_grid, out_img_path):
    original_arr = np.array(original_grid)
    refined_arr = np.array(refined_grid)
    diff_arr = np.abs(original_arr - refined_arr)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(original_arr, cmap='binary', interpolation='nearest')
    ax[0].set_title("Original Grid")
    ax[0].axis('off')
    ax[1].imshow(refined_arr, cmap='binary', interpolation='nearest')
    ax[1].set_title("Refined Grid")
    ax[1].axis('off')
    ax[2].imshow(diff_arr, cmap='gray', interpolation='nearest')
    ax[2].set_title("Difference (Flips)")
    ax[2].axis('off')
    plt.tight_layout()
    plt.savefig(out_img_path)
    plt.close(fig)


########################################
# 10) 批量处理函数，输出结果 CSV
########################################
def batch_refine_and_save_all_weighted(multi_csv="multiple_solutions_details.csv",
                                       out_img_dir="./Greedy_weighted_refined_images",
                                       out_json_dir="./Greedy_weighted_refined_jsons",
                                       max_iterations=50,
                                       time_limit=10.0):
    if not os.path.isfile(multi_csv):
        print(f"错误：找不到 {multi_csv}")
        return
    df = pd.read_csv(multi_csv)
    df_multi = df[df["num_solutions"] > 1]  # 只处理多解的
    if df_multi.empty:
        print("没有多解谜题记录，退出。")
        return
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    if not os.path.exists(out_json_dir):
        os.makedirs(out_json_dir)

    results_list = []
    for idx, row in df_multi.iterrows():
        puzzle_id = row.get("puzzle_id", f"row_{idx}")
        json_path = row["json_file_path"]
        if not os.path.isfile(json_path):
            print(f"文件不存在: {json_path}")
            continue

        print(f"\n===== 处理 Puzzle {puzzle_id} ({json_path}) =====")
        with open(json_path, 'r', encoding='utf-8') as f:
            puzzle_data = json.load(f)
        original_grid = deepcopy(puzzle_data["grid"])

        result = greedy_refinement_weighted(puzzle_data, max_iterations=max_iterations, time_limit=time_limit)
        refined_grid = result["final_grid"]

        out_img_path = os.path.join(out_img_dir, f"puzzle_{puzzle_id}_comparison.png")
        save_comparison_image(original_grid, refined_grid, out_img_path)

        refined_json_path = os.path.join(out_json_dir, f"puzzle_{puzzle_id}_refined.json")
        puzzle_refined_data = {
            "grid": refined_grid,
            "row_hints": compute_row_col_hints(refined_grid)[0],
            "col_hints": compute_row_col_hints(refined_grid)[1]
        }
        with open(refined_json_path, 'w', encoding='utf-8') as jf:
            json.dump(puzzle_refined_data, jf, ensure_ascii=False, indent=2)

        record = {
            "puzzle_id": puzzle_id,
            "json_file_path": json_path,
            "num_solutions": row["num_solutions"],  # 原多解数量
            "iteration_used": result["iteration_used"],
            "time_spent": result["time_spent"],
            "final_hamming_distance": result["final_hamming_distance"],
            "success": result["success"],
            "refined_img_path": out_img_path,
            "refined_json_path": refined_json_path
        }
        results_list.append(record)
        print(
            f"Puzzle {puzzle_id} 完毕: success={record['success']}, iterations={record['iteration_used']}, time={record['time_spent']:.2f}s")

    out_csv = "greedy_weighted_refinement_results.csv"
    df_out = pd.DataFrame(results_list)
    df_out.to_csv(out_csv, index=False)
    print(f"\n=== 所有拼图处理完成，总计 {len(results_list)} 个 ===")
    print(f"结果已保存到 {out_csv}")


########################################
# 11) main
########################################
def main():
    batch_refine_and_save_all_weighted(
        multi_csv="multiple_solutions_details.csv",
        out_img_dir="./Greedy_weighted_refined_images",
        out_json_dir="./Greedy_weighted_refined_jsons",
        max_iterations=50,
        time_limit=10.0
    )


if __name__ == "__main__":
    main()
