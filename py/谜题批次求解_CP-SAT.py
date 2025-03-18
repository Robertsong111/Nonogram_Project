import os
import json
import time
import pandas as pd
from ortools.sat.python import cp_model

# ---------------------------
# CP-SAT 求解器相关函数及回调类
# ---------------------------
class NonogramAllSolutionsCollector(cp_model.CpSolverSolutionCallback):
    """
    遍历 CP-SAT 所有解，把解保存下来。当解数达到 max_solutions 时调用 StopSearch() 停止枚举。
    """
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


def build_multi_block_cp_model(m, n, row_constraints, col_constraints):
    """
    使用 OR-Tools CP-SAT 建模多段 Nonogram 问题：
      row_constraints[i] = [r1, r2, ...] 表示第 i 行的每段黑块长度，
      col_constraints[j] = [c1, c2, ...] 表示第 j 列的每段黑块长度。
    返回 (model, x)，其中 x[i][j] 为布尔变量，表示网格 (i,j) 是否为黑格。
    """
    model = cp_model.CpModel()
    # 定义网格变量 x[i][j]
    x = []
    for i in range(m):
        row_vars = []
        for j in range(n):
            var = model.NewBoolVar(f"x_{i}_{j}")
            row_vars.append(var)
        x.append(row_vars)

    # ---------------------
    # 1) 多段行约束
    # ---------------------
    rowBlockVars = {}
    for i in range(m):
        blocks = row_constraints[i]
        if len(blocks) == 0:
            for j in range(n):
                model.Add(x[i][j] == 0)
            continue
        for b, length_b in enumerate(blocks):
            for start_col in range(n - length_b + 1):
                rowBlockVars[(i, b, start_col)] = model.NewBoolVar(f"rB_{i}_{b}_{start_col}")
        # 每段恰好选一个起始位置
        for b, length_b in enumerate(blocks):
            possible_starts = [rowBlockVars[(i, b, start_col)] for start_col in range(n - length_b + 1)]
            model.Add(sum(possible_starts) == 1)
        # 相邻段之间至少隔 1 列空
        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            for start_s in range(n - length_b + 1):
                for start_t in range(n - blocks[b+1] + 1):
                    if start_t < start_s + length_b + 1:
                        model.Add(rowBlockVars[(i, b, start_s)] + rowBlockVars[(i, b+1, start_t)] <= 1)
        # 每个格子 x[i][col] 与行块变量覆盖关系
        for col in range(n):
            cover_vars = []
            for b, length_b in enumerate(blocks):
                for start_s in range(n - length_b + 1):
                    if start_s <= col < start_s + length_b:
                        cover_vars.append(rowBlockVars[(i, b, start_s)])
            if cover_vars:
                model.Add(x[i][col] <= sum(cover_vars))
                model.Add(x[i][col] >= sum(cover_vars))
            else:
                model.Add(x[i][col] == 0)

    # ---------------------
    # 2) 多段列约束
    # ---------------------
    colBlockVars = {}
    for j in range(n):
        blocks = col_constraints[j]
        if len(blocks) == 0:
            for i in range(m):
                model.Add(x[i][j] == 0)
            continue
        for b, length_b in enumerate(blocks):
            for start_row in range(m - length_b + 1):
                colBlockVars[(j, b, start_row)] = model.NewBoolVar(f"cB_{j}_{b}_{start_row}")
        for b, length_b in enumerate(blocks):
            possible_starts = [colBlockVars[(j, b, start_row)] for start_row in range(m - length_b + 1)]
            model.Add(sum(possible_starts) == 1)
        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            for start_s in range(m - length_b + 1):
                for start_t in range(m - blocks[b+1] + 1):
                    if start_t < start_s + length_b + 1:
                        model.Add(colBlockVars[(j, b, start_s)] + colBlockVars[(j, b+1, start_t)] <= 1)
        for i in range(m):
            cover_vars = []
            for b, length_b in enumerate(blocks):
                for start_s in range(m - length_b + 1):
                    if start_s <= i < start_s + length_b:
                        cover_vars.append(colBlockVars[(j, b, start_s)])
            if cover_vars:
                model.Add(x[i][j] <= sum(cover_vars))
                model.Add(x[i][j] >= sum(cover_vars))
            else:
                model.Add(x[i][j] == 0)

    return model, x


# ---------------------------
# 调用 CP-SAT 求解器，批次处理 JSON 文件
# ---------------------------
def find_all_solutions_cp(m, n, row_constraints, col_constraints, time_limit=10.0, max_solutions=1000):
    """
    调用 build_multi_block_cp_model 构建多段 Nonogram 模型，并使用 CP-SAT 枚举解。
    参数:
      time_limit: 单个谜题求解时间上限（秒）
      max_solutions: 达到此解数后停止搜索
    返回：解的列表（每个解为 m×n 的 0/1 矩阵）
    """
    model, x = build_multi_block_cp_model(m, n, row_constraints, col_constraints)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    collector = NonogramAllSolutionsCollector(x, max_solutions=max_solutions)
    status = solver.SearchForAllSolutions(model, collector)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        print("No solution found or the model is infeasible.")
        return []
    return collector.solutions()


# ---------------------------
# 批量处理 JSON 文件并记录结果（使用 CP-SAT 求解器）
# ---------------------------
def process_puzzle_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def solve_puzzle(puzzle, time_limit=10.0, max_solutions=1000):
    row_hints = puzzle.get("row_hints", puzzle.get("row_constraints"))
    col_hints = puzzle.get("col_hints", puzzle.get("col_constraints"))
    m = len(row_hints)
    n = len(col_hints)
    start = time.time()
    sols = find_all_solutions_cp(m, n, row_hints, col_hints, time_limit, max_solutions)
    end = time.time()
    return end - start, len(sols)


def process_and_solve(file_path, time_limit=10.0, max_solutions=1000):
    try:
        puzzle = process_puzzle_file(file_path)
    except Exception as e:
        return {"puzzle_id": os.path.basename(file_path), "error": f"读取文件出错: {e}"}

    pid = puzzle.get("id", os.path.basename(file_path))
    label = puzzle.get("label", None)
    try:
        solve_time, num_sol = solve_puzzle(puzzle, time_limit, max_solutions)
    except Exception as e:
        return {"puzzle_id": pid, "error": f"求解过程中出错: {e}"}
    return {
        "puzzle_id": pid,
        "label": label,
        "row_hints": str(puzzle.get("row_hints", puzzle.get("row_constraints"))),
        "col_hints": str(puzzle.get("col_hints", puzzle.get("col_constraints"))),
        "num_solutions": num_sol,
        "solving_time_sec": solve_time
    }


def main():
    folder = "./mnist_nonograms"  # 存放谜题 JSON 的文件夹
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".json")]
    total_files = len(files)
    print(f"总共有 {total_files} 个谜题文件。")

    # 对前 10 个谜题采样估计平均求解时间
    sample_size = min(10, total_files)
    sample_times = []
    for file in files[:sample_size]:
        try:
            puzzle = process_puzzle_file(file)
            t, _ = solve_puzzle(puzzle, time_limit=10.0, max_solutions=1000)
            sample_times.append(t)
        except Exception as e:
            print(f"采样文件 {file} 出错: {e}")
    avg_time = sum(sample_times) / sample_size if sample_size > 0 else 0
    estimated_total_time = avg_time * total_files
    print(
        f"基于前 {sample_size} 个谜题的平均求解时间约为 {avg_time:.4f} 秒，预计全部求解时间约为 {estimated_total_time / 60:.2f} 分钟。")

    records = []
    for file in files:
        try:
            record = process_and_solve(file, 10.0, 1000)
        except Exception as e:
            print(f"文件 {file} 处理出错: {e}")
            continue
        if "error" in record:
            print(f"文件 {record['puzzle_id']} 出错: {record['error']}")
        else:
            print(
                f"已处理谜题 {record['puzzle_id']}: {record['num_solutions']} 解, 用时 {record['solving_time_sec']:.4f} 秒")
        records.append(record)

    df = pd.DataFrame(records)
    output_file = "1.csv"
    df.to_csv(output_file, index=False)
    print(f"所有结果已保存至 {output_file}")


if __name__ == "__main__":
    main()