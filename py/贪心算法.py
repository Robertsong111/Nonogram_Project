#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import random
from copy import deepcopy
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import numpy as np
import math

########################################
# 1) 计算行/列提示的辅助函数
########################################
def compute_row_col_hints(grid):
    m = len(grid)
    n = len(grid[0]) if m else 0
    row_hints = []
    for i in range(m):
        row = grid[i]
        hints = []
        count = 0
        for val in row:
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
        hints = []
        count = 0
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
# 2) 建立 Nonogram 模型（同前）
########################################
def build_multi_block_cp_model(m, n, row_constraints, col_constraints):
    model = cp_model.CpModel()
    x = []
    for i in range(m):
        row_vars = []
        for j in range(n):
            var = model.NewBoolVar(f'x_{i}_{j}')
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
                starts.append(model.NewBoolVar(f'rB_{i}_{b}_{start_col}'))
            model.Add(sum(starts) == 1)
            rowBlockVars.append((length_b, starts))
        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            starts_b = rowBlockVars[b][1]
            starts_next = rowBlockVars[b+1][1]
            for s1, var1 in enumerate(starts_b):
                for s2, var2 in enumerate(starts_next):
                    if s2 < s1 + length_b + 1:
                        model.Add(var1 + var2 <= 1)
        for col in range(n):
            cover_exps = []
            for (length_b, starts) in rowBlockVars:
                for s, var_s in enumerate(starts):
                    if s <= col < s + length_b:
                        cover_exps.append(var_s)
            if cover_exps:
                model.Add(x[i][col] == sum(cover_exps))
            else:
                model.Add(x[i][col] == 0)
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
                starts.append(model.NewBoolVar(f'cB_{j}_{b}_{start_row}'))
            model.Add(sum(starts) == 1)
            colBlockVars.append((length_b, starts))
        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            starts_b = colBlockVars[b][1]
            starts_next = colBlockVars[b+1][1]
            for s1, var1 in enumerate(starts_b):
                for s2, var2 in enumerate(starts_next):
                    if s2 < s1 + length_b + 1:
                        model.Add(var1 + var2 <= 1)
        for i in range(m):
            cover_exps = []
            for (length_b, starts) in colBlockVars:
                for s, var_s in enumerate(starts):
                    if s <= i < s + length_b:
                        cover_exps.append(var_s)
            if cover_exps:
                model.Add(x[i][j] == sum(cover_exps))
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
# 4) 数解函数：最多1000
########################################
def count_solutions_up_to_1000(puzzle_data, time_limit=100.0):
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
                    freq[(i,j)] = freq.get((i,j), 0) + 1
    return freq

########################################
# 6A) 连通性检测：使用 DFS 计算黑色连通组件数（8连通）
########################################
def count_black_components(grid):
    m = len(grid)
    if m == 0:
        return 0
    n = len(grid[0])
    visited = [[False]*n for _ in range(m)]
    def dfs(i, j):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= m or cj < 0 or cj >= n:
                continue
            if visited[ci][cj] or grid[ci][cj] == 0:
                continue
            visited[ci][cj] = True
            for di, dj in [(-1,-1), (-1,0), (-1,1),
                           (0,-1),          (0,1),
                           (1,-1),  (1,0),  (1,1)]:
                stack.append((ci+di, cj+dj))
    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1 and not visited[i][j]:
                count += 1
                dfs(i, j)
    return count

########################################
# 6B) 辅助函数 - 计算白色连通组件数
########################################
def count_white_components(grid):
    inverted = [[1 - cell for cell in row] for row in grid]
    return count_black_components(inverted)

########################################
# 6C) 辅助函数 - 检查孤立黑格（针对白→黑翻转）
########################################
def is_isolated_black(grid, i, j):
    m = len(grid)
    n = len(grid[0]) if m > 0 else 0
    directions = [(-1,-1), (-1,0), (-1,1),
                  (0,-1),          (0,1),
                  (1,-1),  (1,0),  (1,1)]
    for di, dj in directions:
        ni, nj = i+di, j+dj
        if 0 <= ni < m and 0 <= nj < n:
            if grid[ni][nj] == 1:
                return False
    return True

########################################
# 6D) 模拟退火微调函数（简单版本）
########################################
def simulated_annealing_refinement(puzzle_data, max_iterations=1000, time_limit=10.0,
                                   T0=20.0, cooling_rate=0.99, w1=1.0, w2=0.01, ambiguity_max_sols=50):
    """
    使用模拟退火进行微调：
      - 状态：当前谜题的 grid；
      - 邻域：随机翻转一个格子（先通过部分解计算歧义格频率，从中按权重选择候选格）；
      - 能量函数：E = w1*(sol_count-1)^2 + w2*diff，其中 diff 为与原图的汉明距离。
    对于翻转：
      - 白→黑：模拟翻转后，若白色连通组件数增加或产生孤立黑格，则拒绝；
      - 黑→白：模拟翻转后，若黑色或白色连通组件数增加，则拒绝。
    当解数为 1 时终止微调。
    返回微调后的 puzzle_data。
    """
    original_grid = deepcopy(puzzle_data.get("grid", []))
    if not original_grid:
        print("警告：puzzle_data 中没有 grid 字段。")
        return puzzle_data

    current_state = deepcopy(original_grid)
    row_h, col_h = compute_row_col_hints(current_state)
    puzzle_data["row_hints"] = row_h
    puzzle_data["col_hints"] = col_h

    # 先计算初始歧义格频率
    m = len(original_grid)
    n = len(original_grid[0]) if m > 0 else 0
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    model, x = build_multi_block_cp_model(m, n, row_h, col_h)
    collector = NonogramAllSolutionsCollector(x, max_solutions=ambiguity_max_sols)
    solver.SearchForAllSolutions(model, collector)
    partial_solutions = collector.solutions()
    ambiguous_freq = compute_ambiguous_cells_with_frequency(partial_solutions, original_grid)
    if not ambiguous_freq:
        print("未检测到歧义格（所有解与原图一致），无需微调。")
        return puzzle_data

    print(f"初始歧义格数量: {len(ambiguous_freq)}")

    def energy(state):
        temp = deepcopy(puzzle_data)
        temp["grid"] = state
        temp["row_hints"], temp["col_hints"] = compute_row_col_hints(state)
        sol_count = count_solutions_up_to_1000(temp, time_limit=time_limit)
        penalty_sol = (sol_count - 1)**2 if sol_count > 1 else 0
        diff = sum(sum(abs(a - b) for a, b in zip(row, orig_row))
                   for row, orig_row in zip(state, original_grid))
        return w1 * penalty_sol + w2 * diff

    current_energy = energy(current_state)
    T = T0

    m_val = len(current_state)
    n_val = len(current_state[0])
    best_count = count_solutions_up_to_1000(puzzle_data, time_limit=time_limit)

    for iteration in range(max_iterations):
        # 检查当前是否还有歧义格
        ambiguous_list = list(ambiguous_freq.keys())
        if not ambiguous_list:
            print("歧义格已为空，终止微调。")
            break
        weights = list(ambiguous_freq.values())
        r, c = random.choices(ambiguous_list, weights=weights, k=1)[0]
        candidate_state = deepcopy(current_state)
        old_val = candidate_state[r][c]
        new_val = 1 - old_val

        # 白→黑翻转：检查翻转后是否增加白色连通组件数或产生孤立黑格
        if new_val == 1:
            temp_state = deepcopy(candidate_state)
            temp_state[r][c] = new_val
            if count_white_components(temp_state) > count_white_components(candidate_state):
                continue
            if is_isolated_black(temp_state, r, c):
                continue

        # 黑→白翻转：检查翻转后是否增加黑色或白色连通组件数
        if new_val == 0:
            temp_state = deepcopy(candidate_state)
            temp_state[r][c] = new_val
            if count_black_components(temp_state) > count_black_components(candidate_state) or \
               count_white_components(temp_state) > count_white_components(candidate_state):
                continue

        candidate_state[r][c] = new_val
        candidate_energy = energy(candidate_state)
        delta_E = candidate_energy - current_energy
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            current_state = candidate_state
            current_energy = candidate_energy
            temp_data = deepcopy(puzzle_data)
            temp_data["grid"] = current_state
            current_count = count_solutions_up_to_1000(temp_data, time_limit=time_limit)
            best_count = current_count
            print(f"Iteration {iteration}: accepted flip at ({r},{c}), energy = {current_energy:.2f}, sol_count = {current_count}")
            if current_count == 1:
                print("达到唯一解，终止微调。")
                break
            # 更新歧义格信息
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = time_limit
            model, x = build_multi_block_cp_model(m_val, n_val, compute_row_col_hints(current_state)[0],
                                                   compute_row_col_hints(current_state)[1])
            collector = NonogramAllSolutionsCollector(x, max_solutions=ambiguity_max_sols)
            solver.SearchForAllSolutions(model, collector)
            partial_solutions = collector.solutions()
            ambiguous_freq = compute_ambiguous_cells_with_frequency(partial_solutions, current_state)
            print(f"更新后的歧义格数量: {len(ambiguous_freq)}")
        T *= cooling_rate

    puzzle_data["grid"] = current_state
    puzzle_data["row_hints"], puzzle_data["col_hints"] = compute_row_col_hints(current_state)
    # 保存对比图到临时文件，不显示
    original_arr = np.array(original_grid)
    refined_arr = np.array(current_state)
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
    plt.savefig("temp_comparison.png")
    plt.close(fig)
    return puzzle_data



########################################
# 7) 批量处理 - 处理 CSV 中所有多解谜题，并保存图片和 refined JSON
########################################
def batch_refine_and_save_all(multi_csv="multiple_solutions_details.csv",
                              out_img_dir="./refined_images",
                              out_json_dir="./refined_jsons",
                              max_iterations=50,
                              time_limit=10.0):
    if not os.path.isfile(multi_csv):
        print(f"错误：找不到 {multi_csv}")
        return
    df = pd.read_csv(multi_csv)
    df_multi = df[df["num_solutions"] > 1]
    if df_multi.empty:
        print("没有多解谜题记录，退出。")
        return
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    if not os.path.exists(out_json_dir):
        os.makedirs(out_json_dir)
    for idx, row in df_multi.iterrows():
        puzzle_id = row.get("puzzle_id", f"row_{idx}")
        json_path = row["json_file_path"]
        if not os.path.isfile(json_path):
            print(f"文件不存在: {json_path}")
            continue
        print(f"\n===== 处理 Puzzle {puzzle_id} ({json_path}) =====")
        with open(json_path, 'r') as f:
            puzzle_data = json.load(f)
        if "grid" in puzzle_data and puzzle_data["grid"]:
            row_h, col_h = compute_row_col_hints(puzzle_data["grid"])
            puzzle_data["row_hints"] = row_h
            puzzle_data["col_hints"] = col_h
        original_grid = deepcopy(puzzle_data["grid"])
        refined_data = simulated_annealing_refinement(puzzle_data, max_iterations=1000,
                                                      time_limit=time_limit, T0=10.0, cooling_rate=0.99,
                                                      w1=1.0, w2=0.01)
        refined_grid = refined_data.get("grid", original_grid)
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
        out_img_path = os.path.join(out_img_dir, f"puzzle_{puzzle_id}_comparison.png")
        plt.savefig(out_img_path)
        plt.close(fig)
        out_json_path = os.path.join(out_json_dir, f"puzzle_{puzzle_id}_refined.json")
        with open(out_json_path, 'w') as jf:
            json.dump(refined_data, jf)
        print(f"Puzzle {puzzle_id} 处理完毕，图像保存在 {out_img_path}，JSON 保存在 {out_json_path}")

def main():
    batch_refine_and_save_all(
        multi_csv="multiple_solutions_details.csv",
        out_img_dir="./refined_images",
        out_json_dir="./refined_jsons",
        max_iterations=50,
        time_limit=100.0
    )

if __name__=="__main__":
    main()
