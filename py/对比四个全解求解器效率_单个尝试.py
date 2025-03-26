import time
import pandas as pd
import matplotlib.pyplot as plt


# 统一接口设计
def solve_with_solver(solver_func, row_constraints, col_constraints):
    """
    统一接口：调用各个求解器并返回求解时间和解的数量
    """
    start_time = time.time()
    solutions = solver_func(row_constraints, col_constraints)
    end_time = time.time()
    solving_time = end_time - start_time
    num_solutions = len(solutions)
    return solving_time, num_solutions


# 求解器 1：CP-SAT 求解器（来自CP_SAT.py）
def solve_with_cp_sat(row_constraints, col_constraints):
    from CP_SAT import find_all_solutions_cp  # 导入CP-SAT的求解函数
    return find_all_solutions_cp(len(row_constraints), len(col_constraints), row_constraints, col_constraints)


# 求解器 2：SAT 求解器（来自SAT.py）
def solve_with_sat(row_constraints, col_constraints):
    from SAT import solve_all_solutions_pure_sat  # 导入SAT的求解函数
    return solve_all_solutions_pure_sat(len(row_constraints), len(col_constraints), row_constraints, col_constraints)


# 求解器 3：时间回溯法求解器（来自论文时间回溯判断.py）
def solve_with_time_backtracking(row_constraints, col_constraints):
    from 论文时间回溯判断 import solve_nonogram  # 导入solve_nonogram函数
    solutions = solve_nonogram(row_constraints, col_constraints)  # 使用时间回溯法进行求解
    return solutions


# 求解器 4：约束传播求解器（来自约束传播.py）
def solve_with_constraint_propagation(row_constraints, col_constraints):
    from 约束传播 import NonogramSolverWithCP  # 导入约束传播求解函数
    solver = NonogramSolverWithCP(row_constraints, col_constraints)
    return solver.solve()


# 将求解器比较并记录结果
def compare_solvers(row_constraints, col_constraints):
    # 记录结果的DataFrame
    results = []

    # 使用不同的求解器
    solvers = [
        ("CP-SAT", solve_with_cp_sat),
        ("SAT", solve_with_sat),
        ("Time Backtracking", solve_with_time_backtracking),
        ("Constraint Propagation", solve_with_constraint_propagation)
    ]

    for solver_name, solver_func in solvers:
        solving_time, num_solutions = solve_with_solver(solver_func, row_constraints, col_constraints)
        results.append({
            "Solver": solver_name,
            "Solving Time (s)": solving_time,
            "Number of Solutions": num_solutions
        })

    # 将结果转为DataFrame
    df = pd.DataFrame(results)

    # 显示结果
    print(df)

    # 绘制图表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制求解时间的柱状图
    ax1.bar(df["Solver"], df["Solving Time (s)"], color='skyblue', label="Solving Time")
    ax1.set_xlabel("Solvers")
    ax1.set_ylabel("Time (s)", color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    # 创建第二个y轴来绘制解的数量
    ax2 = ax1.twinx()
    ax2.plot(df["Solver"], df["Number of Solutions"], color='green', marker='o', label="Number of Solutions")
    ax2.set_ylabel("Number of Solutions", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # 添加标题
    plt.title("Comparison of Nonogram Solvers: Solving Time and Number of Solutions")
    fig.tight_layout()  # 调整布局以防标签重叠
    plt.show()


# 示例调用
if __name__ == "__main__":
    # 示例谜题约束
    row_constraints = [[1], [1], [2], [2], [1], [1]]  # 假设这是一个5x5的谜题行约束
    col_constraints = [[1], [1], [2], [2], [1], [1]]  # 假设这是一个5x5的谜题列约束

    # 调用函数进行比较
    compare_solvers(row_constraints, col_constraints)
