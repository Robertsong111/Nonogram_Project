#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import pandas as pd
import random
from copy import deepcopy
import math
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
import matplotlib.pyplot as plt
import numpy as np
from ortools.sat.python import cp_model


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
# 2. CP-SAT 求解器相关：用于判断唯一性
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
        sol = []
        for i in range(self._m):
            row = []
            for j in range(self._n):
                row.append(self.Value(self._x[i][j]))
            sol.append(row)
        self._solutions.append(sol)
        if len(self._solutions) >= self._max_solutions:
            self.StopSearch()

    def solutions(self):
        return self._solutions


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
            for start in range(n - length_b + 1):
                starts.append(model.NewBoolVar(f"rB_{i}_{b}_{start}"))
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
            for start in range(m - length_b + 1):
                starts.append(model.NewBoolVar(f"cB_{j}_{b}_{start}"))
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
# 3. 连通性检测函数（黑色、白色）以及孤立黑格检查
# ---------------------------
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


# ---------------------------
# 4. MIP 优化方法：基于最小修改使得生成的提示唯一，
#    且保证修改后黑色和白色连通分量不增加（允许减少）
#    并利用歧义格信息对目标函数进行加权
# ---------------------------
def compute_modified_grid(original_grid, delta):
    """
    根据原图和 MIP 模型求解的 delta（0/1矩阵），计算修改后的 grid，
    new_grid[i][j] = original_grid[i][j] XOR delta[i][j]
    """
    m = len(original_grid)
    n = len(original_grid[0])
    new_grid = []
    for i in range(m):
        new_row = []
        for j in range(n):
            new_row.append(original_grid[i][j] ^ delta[i][j])
        new_grid.append(new_row)
    return new_grid


def compute_ambiguity_weight(original_grid):
    """
    利用 SAT 求解器计算原图的歧义格频率，
    并返回一个 m x n 的权重矩阵：weight[i][j] = 1 / (ambiguity + 1)
    意思是歧义频率越高，权重越低，修改该单元格代价较低。
    """
    m = len(original_grid)
    n = len(original_grid[0])
    # 使用 CP-SAT 求解器枚举部分解（上限 50 个）
    row_hints, col_hints = compute_row_col_hints(original_grid)
    puzzle = {"grid": original_grid, "row_hints": row_hints, "col_hints": col_hints}
    sol_count = count_solutions_up_to_1000(puzzle, time_limit=30)
    # 若原图为多解，则计算每个单元格在不同解中出现不同的次数
    # 这里简单地重新求解，并比较每个解与原图的差异
    model, x = build_multi_block_cp_model(m, n, row_hints, col_hints)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    collector = NonogramAllSolutionsCollector(x, max_solutions=50)
    solver.SearchForAllSolutions(model, collector)
    solutions = collector.solutions()
    ambiguous_freq = {}
    for sol in solutions:
        for i in range(m):
            for j in range(n):
                if sol[i][j] != original_grid[i][j]:
                    ambiguous_freq[(i, j)] = ambiguous_freq.get((i, j), 0) + 1
    # 定义权重：权重 = 1 / (freq + 1)
    weight = [[1.0 / (ambiguous_freq.get((i, j), 0) + 1) for j in range(n)] for i in range(m)]
    return weight


def mip_refinement(puzzle_data, mip_time_limit=60, max_mip_iterations=10):
    """
    基于 MIP 求解器对谜题进行最小修改，使得修改后的谜题提示产生唯一解，
    并且保证修改后黑色和白色连通分量不增加（允许减少）。
    这里增加预处理：对于原图中不允许单独翻转的格子（即翻转会破坏连通性），固定为0；
    同时利用歧义格权重信息为目标函数加权。
    """
    original_grid = deepcopy(puzzle_data["grid"])
    m = len(original_grid)
    n = len(original_grid[0])

    # 计算原图的提示及连通分量
    orig_row_hints, orig_col_hints = compute_row_col_hints(original_grid)
    orig_black = count_black_components(original_grid)
    orig_white = count_white_components(original_grid)

    # 预处理：构造 allowed_mask，判断哪些格子允许修改（单独翻转不会增加连通分量）
    allowed_mask = [[1 for j in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            if not allowed_to_flip(original_grid, i, j):
                allowed_mask[i][j] = 0

    # 利用原图的歧义格信息计算权重矩阵（歧义频率越高，权重越低）
    weight_matrix = compute_ambiguity_weight(original_grid)

    # 初始化 MIP 模型
    prob = LpProblem("Nonogram_Modification", LpMinimize)

    # 定义二元变量 delta[i][j]，对于不允许修改的格子固定为0
    delta = [[None for j in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            if allowed_mask[i][j] == 1:
                delta[i][j] = LpVariable(f"delta_{i}_{j}", cat=LpBinary)
            else:
                delta[i][j] = 0

    # 目标函数：加权最小化修改数
    prob += lpSum((weight_matrix[i][j] * delta[i][j]) if allowed_mask[i][j] == 1 else 0
                  for i in range(m) for j in range(n)), "Total_Weighted_Modifications"

    # 切割约束集合
    cuts = []
    best_delta = None

    for iteration in range(max_mip_iterations):
        print(f"MIP 迭代 {iteration} 开始求解...")
        status = prob.solve(PULP_CBC_CMD(timeLimit=mip_time_limit, msg=0))
        if status != 1:
            print("MIP 求解失败或超时。")
            break

        # 提取 MIP 解
        candidate_delta = [[int(delta[i][j].varValue) if allowed_mask[i][j] == 1 else 0
                            for j in range(n)] for i in range(m)]
        candidate_grid = compute_modified_grid(original_grid, candidate_delta)
        new_row_hints, new_col_hints = compute_row_col_hints(candidate_grid)
        candidate_puzzle = {
            "grid": candidate_grid,
            "row_hints": new_row_hints,
            "col_hints": new_col_hints
        }
        sol_count = count_solutions_up_to_1000(candidate_puzzle, time_limit=30)
        cand_black = count_black_components(candidate_grid)
        cand_white = count_white_components(candidate_grid)
        mod_count = sum(sum(row) for row in candidate_delta)
        print(f"当前修改 {mod_count} 个格子, SAT 检测解数为 {sol_count}")
        print(f"原图黑连通分量 {orig_black}，新图 {cand_black}；原图白连通分量 {orig_white}，新图 {cand_white}")

        if sol_count == 1 and cand_black <= orig_black and cand_white <= orig_white:
            print("找到满足唯一性和连通性要求的修改方案。")
            best_delta = candidate_delta
            puzzle_data["grid"] = candidate_grid
            puzzle_data["row_hints"] = new_row_hints
            puzzle_data["col_hints"] = new_col_hints
            return puzzle_data
        else:
            # 添加切割约束：排除当前 candidate
            cut_expr = lpSum(delta[i][j] if candidate_delta[i][j] == 1 else (1 - delta[i][j])
                             for i in range(m) for j in range(n) if allowed_mask[i][j] == 1)
            total_allowed = sum(allowed_mask[i][j] for i in range(m) for j in range(n))
            prob += cut_expr <= total_allowed - 1, f"Cut_{iteration}"
            cuts.append(cut_expr)
            print(f"添加切割约束，当前切割约束数：{len(cuts)}")
    print("MIP 迭代达到最大次数，但未能找到满足要求的修改方案。")
    return puzzle_data


def allowed_to_flip(original_grid, i, j):
    """
    判断单独翻转原图中 (i,j) 这个格子是否会导致黑色或白色连通分量增加。
    如果翻转后连通分量增加，则返回 False。
    """
    orig_black = count_black_components(original_grid)
    orig_white = count_white_components(original_grid)
    candidate = deepcopy(original_grid)
    candidate[i][j] = 1 - candidate[i][j]
    new_black = count_black_components(candidate)
    new_white = count_white_components(candidate)
    if new_black > orig_black or new_white > orig_white:
        return False
    return True


# ---------------------------
# 5. 保存对比图函数
# ---------------------------
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


# ---------------------------
# 6. 批量处理 MIP 微调：结合现有数据批量处理谜题，
#    同时保存微调后 JSON 和对比图片
# ---------------------------
def batch_mip_refine_and_save(multi_csv="multiple_solutions_details.csv",
                              out_json_dir="./mip_refined_jsons",
                              out_img_dir="./mip_refined_images",
                              mip_time_limit=60,
                              max_mip_iterations=10):
    if not os.path.isfile(multi_csv):
        print(f"错误：找不到 {multi_csv}")
        return
    df = pd.read_csv(multi_csv)
    df_multi = df[df["num_solutions"] > 1]
    if df_multi.empty:
        print("没有多解谜题记录，退出。")
        return
    if not os.path.exists(out_json_dir):
        os.makedirs(out_json_dir)
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    for idx, row in df_multi.iterrows():
        puzzle_id = row.get("puzzle_id", f"row_{idx}")
        json_path = row["json_file_path"]
        if not os.path.isfile(json_path):
            print(f"文件不存在: {json_path}")
            continue
        print(f"\n===== 处理 Puzzle {puzzle_id} ({json_path}) =====")
        with open(json_path, 'r', encoding='utf-8') as f:
            puzzle_data = json.load(f)
        if "grid" in puzzle_data and puzzle_data["grid"]:
            r_h, c_h = compute_row_col_hints(puzzle_data["grid"])
            puzzle_data["row_hints"] = r_h
            puzzle_data["col_hints"] = c_h
        original_grid = deepcopy(puzzle_data["grid"])
        refined_data = mip_refinement(puzzle_data, mip_time_limit=mip_time_limit, max_mip_iterations=max_mip_iterations)
        refined_grid = refined_data.get("grid", original_grid)
        # 保存对比图片
        out_img_path = os.path.join(out_img_dir, f"puzzle_{puzzle_id}_comparison.png")
        save_comparison_image(original_grid, refined_grid, out_img_path)
        # 保存微调后的 JSON
        out_json_path = os.path.join(out_json_dir, f"puzzle_{puzzle_id}_mip_refined.json")
        with open(out_json_path, 'w', encoding='utf-8') as jf:
            json.dump(refined_data, jf, ensure_ascii=False, indent=2)
        print(f"Puzzle {puzzle_id} 微调完成，图片保存在 {out_img_path}，JSON 保存在 {out_json_path}")


def main():
    batch_mip_refine_and_save(multi_csv="multiple_solutions_details.csv",
                              out_json_dir="./mip_refined_jsons",
                              out_img_dir="./mip_refined_images",
                              mip_time_limit=60,
                              max_mip_iterations=50)


if __name__ == "__main__":
    main()
