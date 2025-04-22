import time
from pysat.solvers import Glucose4
from pysat.formula import CNF, IDPool
import matplotlib.pyplot as plt
def encode_multi_block_nonogram_to_cnf(m, n, row_constraints, col_constraints):
    """
    将“行多段 + 列多段”的 Nonogram 约束编码为 CNF。
    返回 (cnf, varmap):
      - cnf: pysat.formula.CNF 对象，包含所有子句
      - varmap: 包含:
          varmap['x'][(i,j)]        -> int (变量ID), 表示 x[i][j]
          varmap['rblock'][(i,b,s)] -> int (变量ID), 表示第 i 行的第 b 段从列 s 开始
          varmap['cblock'][(j,b,s)] -> int (变量ID), 表示第 j 列的第 b 段从行 s 开始
    """

    pool = IDPool()     # 用来统一管理变量ID
    cnf = CNF()         # 存放所有子句
    varmap = {
        'x': {},
        'rblock': {},
        'cblock': {}
    }

    # 1) x[i][j] 变量
    for i in range(m):
        for j in range(n):
            name = f"x_{i}_{j}"
            var_id = pool.id(name)
            varmap['x'][(i, j)] = var_id

    # 2) 行块变量 rblock[(i,b,start)]
    #    row_constraints[i] = [l1, l2, ...] 可能多段
    #    b 表示第 b 段, start 表示该段的起始列
    for i in range(m):
        blocks = row_constraints[i]
        for b, length_b in enumerate(blocks):
            for start_col in range(n - length_b + 1):
                name = f"rB_{i}_{b}_{start_col}"
                var_id = pool.id(name)
                varmap['rblock'][(i, b, start_col)] = var_id

    # 3) 列块变量 cblock[(j,b,start)]
    for j in range(n):
        blocks = col_constraints[j]
        for b, length_b in enumerate(blocks):
            for start_row in range(m - length_b + 1):
                name = f"cB_{j}_{b}_{start_row}"
                var_id = pool.id(name)
                varmap['cblock'][(j, b, start_row)] = var_id

    # -----------------------------
    # 开始添加子句 (行约束)
    # -----------------------------
    for i in range(m):
        blocks = row_constraints[i]
        # 若无黑段 => 这一行全是白(0)
        if len(blocks) == 0:
            # x[i][j] = 0 => -x_id
            for j in range(n):
                x_id = varmap['x'][(i, j)]
                cnf.append([-x_id])
            continue

        # (A) 每段 exactly one 起始位置
        for b, length_b in enumerate(blocks):
            # possible_starts = 所有 start_col
            possible_ids = []
            for start_col in range(n - length_b + 1):
                rblock_id = varmap['rblock'][(i, b, start_col)]
                possible_ids.append(rblock_id)
            # (A1) 至少 1 => OR
            cnf.append(possible_ids)
            # (A2) 两两不同时真 => pairwise (-a, -b)
            for a in range(len(possible_ids)):
                for c in range(a+1, len(possible_ids)):
                    cnf.append([-possible_ids[a], -possible_ids[c]])

        # (B) 相邻段要至少隔 1 列空
        #     若段 b 从 start_s 开始, 段 b+1 从 start_t 开始
        #     则若 start_t < start_s + length_b + 1, 不能同时为真
        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            for start_s in range(n - length_b + 1):
                rblock_s = varmap['rblock'][(i, b, start_s)]
                for start_t in range(n - blocks[b+1] + 1):
                    if start_t < start_s + length_b + 1:
                        # => -rblock_s OR -rblock_t
                        rblock_t = varmap['rblock'][(i, b+1, start_t)]
                        cnf.append([-rblock_s, -rblock_t])

        # (C) coverage: x[i][col] = 1 当且仅当 它被某个行段覆盖
        #     “被某个行段覆盖” = ∨_{(b,start_s) 覆盖col} rblock[i,b,start_s]
        #     由于可多段, x[i][col] 可能由其中1段覆盖(不允许段重叠到同一格)
        for col in range(n):
            cover_ids = []
            for b, length_b in enumerate(blocks):
                for start_s in range(n - length_b + 1):
                    # 若 col 在 [start_s, start_s+length_b-1]
                    if start_s <= col < start_s + length_b:
                        cover_ids.append(varmap['rblock'][(i, b, start_s)])

            x_id = varmap['x'][(i, col)]
            if not cover_ids:
                # 这一格无法被任何段覆盖 => x[i][col] = 0
                cnf.append([-x_id])
            else:
                # x_id => (cover_ids 之一为真)
                # => (-x_id OR cover_ids[0] OR cover_ids[1] ...)
                cnf.append([-x_id] + cover_ids)
                # 反向: 任意 cover_ids[k] => x_id
                for c_id in cover_ids:
                    cnf.append([-c_id, x_id])

    # -----------------------------
    # 列约束 (同理)
    # -----------------------------
    for j in range(n):
        blocks = col_constraints[j]
        if len(blocks) == 0:
            # 整列都是白
            for i in range(m):
                x_id = varmap['x'][(i, j)]
                cnf.append([-x_id])
            continue

        # (A) exactly one start
        for b, length_b in enumerate(blocks):
            possible_ids = []
            for start_i in range(m - length_b + 1):
                cid = varmap['cblock'][(j, b, start_i)]
                possible_ids.append(cid)
            # OR
            cnf.append(possible_ids)
            # pairwise not both true
            for a in range(len(possible_ids)):
                for c in range(a+1, len(possible_ids)):
                    cnf.append([-possible_ids[a], -possible_ids[c]])

        # (B) 相邻列段至少隔 1 行
        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            for start_s in range(m - length_b + 1):
                cblock_s = varmap['cblock'][(j, b, start_s)]
                for start_t in range(m - blocks[b+1] + 1):
                    if start_t < start_s + length_b + 1:
                        cblock_t = varmap['cblock'][(j, b+1, start_t)]
                        cnf.append([-cblock_s, -cblock_t])

        # (C) coverage: x[i][j] = 1 当且仅当 被某个列段覆盖
        for i in range(m):
            cover_ids = []
            for b, length_b in enumerate(blocks):
                for start_s in range(m - length_b + 1):
                    if start_s <= i < start_s + length_b:
                        cover_ids.append(varmap['cblock'][(j, b, start_s)])
            x_id = varmap['x'][(i, j)]
            if not cover_ids:
                cnf.append([-x_id])
            else:
                cnf.append([-x_id] + cover_ids)
                for c_id in cover_ids:
                    cnf.append([-c_id, x_id])

    return cnf, varmap

def decode_solution(m, n, varmap, model_solution):
    """
    将SAT解映射回 0/1 矩阵 (x[i][j])。
    model_solution: dict(var_id -> bool), 表示该变量是否为 True。
    """
    sol_matrix = []
    for i in range(m):
        rowvals = []
        for j in range(n):
            x_id = varmap['x'][(i, j)]
            val = model_solution[x_id]
            rowvals.append(1 if val else 0)
        sol_matrix.append(rowvals)
    return sol_matrix

def add_blocking_clause(cnf, varmap, solution_matrix):
    """
    对已知解添加阻断子句 (blocking clause)，以排除该解，再继续求解。
    对每个 x[i][j]:
      若 solution_matrix[i][j] = 1 => 加 -x[i][j]
      若 solution_matrix[i][j] = 0 => 加 x[i][j]
    这些文字 OR 在一起 => CNF 里再加一条子句。
    """
    clause = []
    m = len(solution_matrix)
    n = len(solution_matrix[0]) if m > 0 else 0
    for i in range(m):
        for j in range(n):
            val = solution_matrix[i][j]
            x_id = varmap['x'][(i, j)]
            if val == 1:
                clause.append(-x_id)
            else:
                clause.append(x_id)
    cnf.append(clause)

def solve_all_solutions_pure_sat(m, n, row_constraints, col_constraints):
    """
    纯 SAT 构建 + 枚举所有可行解（多段 Nonogram）。
    返回所有解的列表，每个解是 m×n 的0/1矩阵。
    """
    cnf, varmap = encode_multi_block_nonogram_to_cnf(m, n, row_constraints, col_constraints)

    solutions = []

    while True:
        solver = Glucose4()
        # 加载所有子句
        for clause in cnf.clauses:
            solver.add_clause(clause)

        ok = solver.solve()
        if not ok:
            # 无解/不可行 => 停止
            solver.delete()
            break

        # 读取解
        model = solver.get_model()  # list of (var or -var)
        assignment = {}
        for lit in model:
            if lit > 0:
                assignment[lit] = True
            else:
                assignment[-lit] = False

        sol_matrix = decode_solution(m, n, varmap, assignment)
        solutions.append(sol_matrix)

        # 添加阻断子句，排除该解
        add_blocking_clause(cnf, varmap, sol_matrix)

        solver.delete()

    return solutions


# ------------------------------------------------------------------------------------------
# 以上是“纯 SAT 多段 Nonogram” 的完整实现。下面给一个示例主程序，可与 CP-SAT 版本对比。
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # 这里演示一个较小规模的例子 (5x5 或 6x6)，方便测试。
    # row_constraints[i] / col_constraints[j] = [段1长度, 段2长度, ...]
    #
    # 例如和你之前 CP 代码里那种 row_constraints = [ [4], [1,1], [1,1], [1,1], [1,1], ... ]
    # 如果你想做 10x10，可以直接改大，但枚举所有解会非常耗时，需谨慎。
    import sys
    sys.setrecursionlimit(10**7)

    # 举例: 5行×5列
    row_constraints = [[1], [1], [1], [1], [1], [1], [1]]
    col_constraints = [[1], [1], [1], [1], [1], [1], [1]]
    m = len(row_constraints)
    n = len(col_constraints)

    print("使用纯 SAT 方式，开始枚举所有解...")
    start_t = time.time()
    all_solutions = solve_all_solutions_pure_sat(m, n, row_constraints, col_constraints)
    end_t = time.time()

    print(f"共找到 {len(all_solutions)} 个可行解，用时 {end_t - start_t:.8f} 秒，SAT求解器计算.")
    for idx, sol in enumerate(all_solutions[:3], start=1):  # 如需展示更多改 [:3]
        plt.imshow(sol, cmap="gray_r", interpolation="none")
        plt.axis("off")
        plt.title(f"Solution #{idx}")
        plt.show()
