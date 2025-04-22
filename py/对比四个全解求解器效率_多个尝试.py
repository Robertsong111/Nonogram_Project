import os
import json
import time
import random
import pandas as pd
import matplotlib.pyplot as plt


# 统一接口设计
def solve_with_solver(solver_func, row_constraints, col_constraints):
    """
    统一接口：调用各个求解器并返回求解时间和解的数量
    """
    start_time = time.time()
    solutions = solver_func(row_constraints, col_constraints)  # 去掉解数量限制
    end_time = time.time()
    solving_time = end_time - start_time
    num_solutions = len(solutions)
    return solving_time, num_solutions


# 求解器 1：CP-SAT 求解器（来自CP_SAT.py）
def solve_with_cp_sat(row_constraints, col_constraints):
    from CP_SAT import find_all_solutions_cp  # 导入CP-SAT的求解函数
    print("正在使用CP-SAT求解...")
    solutions = find_all_solutions_cp(len(row_constraints), len(col_constraints), row_constraints, col_constraints)
    print(f"CP-SAT求解完成，找到 {len(solutions)} 个解")
    return solutions


# 求解器 2：SAT 求解器（来自SAT.py）
def solve_with_sat(row_constraints, col_constraints):
    from SAT import solve_all_solutions_pure_sat  # 导入SAT的求解函数
    print("正在使用SAT求解...")
    solutions = solve_all_solutions_pure_sat(len(row_constraints), len(col_constraints), row_constraints,
                                             col_constraints)
    print(f"SAT求解完成，找到 {len(solutions)} 个解")
    return solutions


# 求解器 3：时间回溯法求解器（来自论文时间回溯判断.py）
def solve_with_time_backtracking(row_constraints, col_constraints):
    from 论文时间回溯判断 import solve_nonogram  # 导入solve_nonogram函数
    print("正在使用时间回溯法求解...")
    solutions = solve_nonogram(row_constraints, col_constraints)  # 使用时间回溯法进行求解
    print(f"时间回溯法求解完成，找到 {len(solutions)} 个解")
    return solutions


# 求解器 4：约束传播求解器（来自约束传播.py）
def solve_with_constraint_propagation(row_constraints, col_constraints):
    from 约束传播 import NonogramSolverWithCP  # 导入约束传播求解函数
    print("正在使用约束传播求解...")
    solver = NonogramSolverWithCP(row_constraints, col_constraints)
    solutions = solver.solve()
    print(f"约束传播求解完成，找到 {len(solutions)} 个解")
    return solutions


# 从生成的JSON文件中读取谜题并进行求解
def process_puzzle_files(puzzle_dir, max_puzzles=50):
    puzzle_files = [os.path.join(puzzle_dir, f) for f in os.listdir(puzzle_dir) if f.lower().endswith('.json')]

    # 随机选择max_puzzles个谜题
    puzzle_files = random.sample(puzzle_files, min(len(puzzle_files), max_puzzles))  # 随机选择最多max_puzzles个文件

    all_results = []

    for puzzle_file in puzzle_files:
        # 读取谜题数据
        with open(puzzle_file, 'r') as f:
            puzzle_data = json.load(f)

        row_constraints = puzzle_data['row_constraints']
        col_constraints = puzzle_data['col_constraints']

        results = {}

        # 使用不同的求解器
        solvers = [
            ("CP-SAT", solve_with_cp_sat),
            #("SAT", solve_with_sat),
            #("Time Backtracking", solve_with_time_backtracking),
            #("Constraint Propagation", solve_with_constraint_propagation)
        ]

        for solver_name, solver_func in solvers:
            solving_time, num_solutions = solve_with_solver(solver_func, row_constraints, col_constraints)
            results[solver_name] = {
                "Solving Time (s)": solving_time,
                "Number of Solutions": num_solutions
            }

        all_results.append({
            "Puzzle ID": puzzle_data["id"],
            "Results": results
        })

    return all_results


# 将求解器比较并记录结果
def compare_solvers(puzzle_dir, max_puzzles=50):
    all_results = process_puzzle_files(puzzle_dir, max_puzzles)

    # 记录每个求解器的总时间和解的数量
    solver_stats = {
        "Solver": [],
        "Avg Solving Time (s)": [],
        "Avg Number of Solutions": []
    }

    # 使用不同的求解器统计结果
    solvers = ["CP-SAT", "SAT", "Time Backtracking", "Constraint Propagation"]

    for solver_name in solvers:
        total_time = 0
        total_solutions = 0
        puzzle_count = 0

        for result in all_results:
            puzzle_results = result["Results"]
            if solver_name in puzzle_results:
                total_time += puzzle_results[solver_name]["Solving Time (s)"]
                total_solutions += puzzle_results[solver_name]["Number of Solutions"]
                puzzle_count += 1

        # 计算平均时间和平均解数
        avg_time = total_time / puzzle_count if puzzle_count > 0 else 0
        avg_solutions = total_solutions / puzzle_count if puzzle_count > 0 else 0

        solver_stats["Solver"].append(solver_name)
        solver_stats["Avg Solving Time (s)"].append(avg_time)
        solver_stats["Avg Number of Solutions"].append(avg_solutions)

    # 将结果转为DataFrame
    df = pd.DataFrame(solver_stats)

    # 显示结果
    print(df)

    # 绘制图表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制求解时间的柱状图
    bars = ax1.bar(df["Solver"], df["Avg Solving Time (s)"], color='skyblue', label="Solving Time")
    ax1.set_xlabel("Solvers")
    ax1.set_ylabel("Avg Time (s)", color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    # 在每个柱子上显示具体数字
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom', color='black')

    # 创建第二个y轴来绘制解的数量
    ax2 = ax1.twinx()
    ax2.plot(df["Solver"], df["Avg Number of Solutions"], color='green', marker='o', label="Avg Number of Solutions")
    ax2.set_ylabel("Avg Number of Solutions", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # 添加标题
    plt.title("Comparison of 5*5 Nonogram Solvers: Average Solving Time and Number of Solutions")

    # 调整布局，防止标签重叠
    plt.subplots_adjust(bottom=0.15, top=0.85)  # 调整上下边距

    # 显示图表
    plt.show()


# 示例调用
if __name__ == "__main__":
    puzzle_dir = "./mnist_nonograms"  # 设定谜题存储的目录路径
    compare_solvers(puzzle_dir, max_puzzles=5000)  # 随机挑选50个谜题并进行比较
