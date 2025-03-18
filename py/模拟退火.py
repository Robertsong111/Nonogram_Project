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

# ---------------------------
# 1. 辅助函数：计算行/列提示
# ---------------------------
def compute_row_col_hints(grid):
    m = len(grid)
    n = len(grid[0]) if m else 0
    row_hints = []
    for i in range(m):
        hint = []
        count = 0
        for val in grid[i]:
            if val == 1:
                count += 1
            else:
                if count:
                    hint.append(count)
                    count = 0
        if count:
            hint.append(count)
        row_hints.append(hint)
    col_hints = []
    for j in range(n):
        hint = []
        count = 0
        for i in range(m):
            if grid[i][j] == 1:
                count += 1
            else:
                if count:
                    hint.append(count)
                    count = 0
        if count:
            hint.append(count)
        col_hints.append(hint)
    return row_hints, col_hints

# ---------------------------
# 2. CP-SAT 建模：构造 Nonogram 模型
# ---------------------------
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
                starts.append(model.NewBoolVar(f"cB_{j}_{b}_{start_row}"))
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

# ---------------------------
# 3. CP-SAT 解回调：收集所有解（或上限数量）
# ---------------------------
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

# ---------------------------
# 4. 统计解数：最多统计 1000 个解
# ---------------------------
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

# ---------------------------
# 5. 连通性检测及孤立黑格判断
# ---------------------------
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
                           (1,-1), (1,0), (1,1)]:
                stack.append((ci+di, cj+dj))
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
    directions = [(-1,-1), (-1,0), (-1,1),
                  (0,-1),          (0,1),
                  (1,-1),  (1,0),  (1,1)]
    for di, dj in directions:
        ni, nj = i+di, j+dj
        if 0 <= ni < m and 0 <= nj < n:
            if grid[ni][nj] == 1:
                return False
    return True

# ---------------------------
# 6. 模拟退火微调函数：将多解谜题微调为唯一解
# ---------------------------
def simulated_annealing_refinement(puzzle_data, max_iterations=1000, time_limit=10.0,
                                   T0=20.0, cooling_rate=0.99, w1=1.0, w2=0.01, ambiguity_max_sols=50):
    """
    利用模拟退火对多解谜题进行微调，目标是使谜题达到唯一解，同时保持与原始谜题差异较小。
    参数说明：
      puzzle_data: 包含 "grid" 以及行/列提示（如不存在则自动生成）。
      max_iterations: 最大迭代次数。
      time_limit: 每次求解的时间上限（秒）。
      T0: 初始温度。
      cooling_rate: 降温速率。
      w1: 解数惩罚项权重（使解数越多能量越高）。
      w2: 与原始 grid 差异惩罚权重。
      ambiguity_max_sols: 统计歧义格时枚举的最大解数。
    """
    original_grid = deepcopy(puzzle_data["grid"])
    current_state = deepcopy(original_grid)
    # 如果提示不存在，计算并更新
    if "row_hints" not in puzzle_data or "col_hints" not in puzzle_data:
        row_h, col_h = compute_row_col_hints(current_state)
        puzzle_data["row_hints"] = row_h
        puzzle_data["col_hints"] = col_h

    # 计算初始能量
    current_sol_count = count_solutions_up_to_1000(puzzle_data, time_limit)
    current_diff = sum(sum(1 for a, b in zip(r1, r2) if a != b) for r1, r2 in zip(current_state, original_grid))
    current_energy = w1 * (current_sol_count - 1)**2 + w2 * current_diff
    T = T0

    for iteration in range(max_iterations):
        m_val = len(current_state)
        n_val = len(current_state[0]) if m_val > 0 else 0
        # 枚举部分解，统计歧义格频率
        temp_data = {"grid": current_state, "row_hints": puzzle_data["row_hints"], "col_hints": puzzle_data["col_hints"]}
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        model, x = build_multi_block_cp_model(m_val, n_val, temp_data["row_hints"], temp_data["col_hints"])
        collector = NonogramAllSolutionsCollector(x, max_solutions=ambiguity_max_sols)
        solver.SearchForAllSolutions(model, collector)
        partial_solutions = collector.solutions()
        ambiguous_freq = {}
        for sol in partial_solutions:
            for i in range(m_val):
                for j in range(n_val):
                    if sol[i][j] != current_state[i][j]:
                        ambiguous_freq[(i, j)] = ambiguous_freq.get((i, j), 0) + 1
        if not ambiguous_freq:
            print("无歧义格，提前结束退火。")
            break

        # 根据歧义频率加权随机选择一个格子进行翻转
        cells = list(ambiguous_freq.keys())
        weights = list(ambiguous_freq.values())
        r, c = random.choices(cells, weights=weights, k=1)[0]

        candidate_state = deepcopy(current_state)
        candidate_state[r][c] = 1 - candidate_state[r][c]

        # 连通性检查：白→黑时，若增加白色连通组件或产生孤立黑格则拒绝；黑→白时，若增加黑/白连通组件则拒绝
        if candidate_state[r][c] == 1:  # 白变黑
            if count_white_components(candidate_state) > count_white_components(current_state):
                continue
            if is_isolated_black(candidate_state, r, c):
                continue
        else:  # 黑变白
            if count_black_components(candidate_state) > count_black_components(current_state) or \
               count_white_components(candidate_state) > count_white_components(current_state):
                continue

        # 更新候选状态提示
        cand_row_h, cand_col_h = compute_row_col_hints(candidate_state)
        candidate_data = {"grid": candidate_state, "row_hints": cand_row_h, "col_hints": cand_col_h}
        candidate_sol_count = count_solutions_up_to_1000(candidate_data, time_limit)
        candidate_diff = sum(sum(1 for a, b in zip(r1, r2) if a != b) for r1, r2 in zip(candidate_state, original_grid))
        candidate_energy = w1 * (candidate_sol_count - 1)**2 + w2 * candidate_diff

        delta_E = candidate_energy - current_energy
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            current_state = candidate_state
            current_energy = candidate_energy
            current_sol_count = candidate_sol_count
            print(f"Iteration {iteration}: accepted flip at ({r},{c}), energy = {current_energy:.2f}, sol_count = {current_sol_count}")
            if current_sol_count == 1:
                print("唯一解达到，终止退火。")
                break
        T *= cooling_rate

    # 更新 puzzle_data 中的 grid 与提示
    puzzle_data["grid"] = current_state
    new_row_h, new_col_h = compute_row_col_hints(current_state)
    puzzle_data["row_hints"] = new_row_h
    puzzle_data["col_hints"] = new_col_h
    return puzzle_data

# ---------------------------
# 7. 批量处理：读取 CSV 中多解谜题，微调后保存 JSON 和对比图
# ---------------------------
def batch_refine_and_save_all(multi_csv="multiple_solutions_details.csv",
                              out_img_dir="./refined_images",
                              out_json_dir="./refined_jsons",
                              max_iterations=500,
                              time_limit=10.0):
    if not os.path.isfile(multi_csv):
        print(f"错误：找不到 {multi_csv}")
        return
    df = pd.read_csv(multi_csv)
    # 筛选多解谜题记录
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
            r_h, c_h = compute_row_col_hints(puzzle_data["grid"])
            puzzle_data["row_hints"] = r_h
            puzzle_data["col_hints"] = c_h
        original_grid = deepcopy(puzzle_data["grid"])
        refined_data = simulated_annealing_refinement(puzzle_data, max_iterations=max_iterations,
                                                      time_limit=time_limit, T0=10.0, cooling_rate=0.99,
                                                      w1=1.0, w2=0.05)
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
            json.dump(refined_data, jf, ensure_ascii=False, indent=2)
        print(f"Puzzle {puzzle_id} 处理完毕，图像保存在 {out_img_path}，JSON 保存在 {out_json_path}")

# ---------------------------
# 8. 主函数入口
# ---------------------------
def main():
    # 请确保 CSV 文件 "multiple_solutions_details.csv" 存在，且其中包含 "puzzle_id" 与 "json_file_path" 字段
    batch_refine_and_save_all(multi_csv="multiple_solutions_details.csv",
                              out_img_dir="./模拟退火_refined_images",
                              out_json_dir="./模拟退火_refined_jsons",
                              max_iterations=500,
                              time_limit=10.0)

if __name__ == "__main__":
    main()
