import copy
import time
import matplotlib.pyplot as plt

def generate_line_configurations(constraint, length):
    results = []
    def backtrack(current, idx, cidx):
        if cidx == len(constraint):
            if idx <= length:
                new_line = current + [0]*(length-len(current))
                results.append(new_line)
            return
        max_start = length - sum(constraint[cidx:]) - (len(constraint) - cidx - 1)
        for start_pos in range(idx, max_start+1):
            new_line = current + [0]*(start_pos-len(current)) + [1]*constraint[cidx]
            if cidx < len(constraint) - 1:
                new_line.append(0)
            backtrack(new_line, len(new_line), cidx+1)
    backtrack([], 0, 0)
    return results

def transpose(matrix):
    return list(map(list, zip(*matrix)))

class NonogramSolverWithCP:
    def __init__(self, row_constraints, col_constraints):
        self.m = len(row_constraints)
        self.n = len(col_constraints)
        self.row_constraints = row_constraints
        self.col_constraints = col_constraints
        self.row_poss = [generate_line_configurations(rc, self.n) for rc in self.row_constraints]
        self.col_poss = [generate_line_configurations(cc, self.m) for cc in self.col_constraints]
        self.solutions = []

    def propagate_constraints(self, row_poss, col_poss):
        updated = False
        for i in range(self.m):
            new_row_configurations = []
            for row_cfg in row_poss[i]:
                compatible = all(any(col_cfg[i] == row_cfg[j] for col_cfg in col_poss[j]) for j in range(self.n))
                if compatible:
                    new_row_configurations.append(row_cfg)
            if len(new_row_configurations) < len(row_poss[i]):
                updated = True
            row_poss[i] = new_row_configurations
        for j in range(self.n):
            new_col_configurations = []
            for col_cfg in col_poss[j]:
                compatible = all(any(row_cfg[j] == col_cfg[i] for row_cfg in row_poss[i]) for i in range(self.m))
                if compatible:
                    new_col_configurations.append(col_cfg)
            if len(new_col_configurations) < len(col_poss[j]):
                updated = True
            col_poss[j] = new_col_configurations
        return row_poss, col_poss, updated

    def is_consistent(self, row_poss, col_poss):
        return all(row_poss[i] for i in range(self.m)) and all(col_poss[j] for j in range(self.n))

    def solve_logic(self, row_poss, col_poss):
        while True:
            row_poss, col_poss, updated = self.propagate_constraints(row_poss, col_poss)
            if not self.is_consistent(row_poss, col_poss):
                return
            if not updated:
                break
        if all(len(x) == 1 for x in row_poss) and all(len(x) == 1 for x in col_poss):
            solution_board = [row_poss[i][0] for i in range(self.m)]
            if self.verify_solution(solution_board):
                self.solutions.append(solution_board)
            return
        row_idx, row_min = None, float('inf')
        for i in range(self.m):
            l = len(row_poss[i])
            if 1 < l < row_min:
                row_min = l
                row_idx = i
        col_idx, col_min = None, float('inf')
        for j in range(self.n):
            l = len(col_poss[j])
            if 1 < l < col_min:
                col_min = l
                col_idx = j
        if row_idx is not None or col_idx is not None:
            if row_min <= col_min:
                for cfg in row_poss[row_idx]:
                    new_row_poss = copy.deepcopy(row_poss)
                    new_col_poss = copy.deepcopy(col_poss)
                    new_row_poss[row_idx] = [cfg]
                    self.solve_logic(new_row_poss, new_col_poss)
            else:
                for cfg in col_poss[col_idx]:
                    new_row_poss = copy.deepcopy(row_poss)
                    new_col_poss = copy.deepcopy(col_poss)
                    new_col_poss[col_idx] = [cfg]
                    self.solve_logic(new_row_poss, new_col_poss)

    def verify_solution(self, board):
        for i in range(self.m):
            if not self.check_line(board[i], self.row_constraints[i]):
                return False
        for j in range(self.n):
            if not self.check_line([board[i][j] for i in range(self.m)], self.col_constraints[j]):
                return False
        return True

    def check_line(self, line, constraint):
        blocks, cur = [], 0
        for cell in line:
            if cell == 1:
                cur += 1
            elif cur > 0:
                blocks.append(cur)
                cur = 0
        if cur > 0:
            blocks.append(cur)
        return blocks == constraint

    def solve(self):
        row_poss_cpy = copy.deepcopy(self.row_poss)
        col_poss_cpy = copy.deepcopy(self.col_poss)
        self.solve_logic(row_poss_cpy, col_poss_cpy)
        return self.solutions

def display_solutions(solutions, max_display=5):
    for idx, sol in enumerate(solutions[:max_display]):
        plt.imshow(sol, cmap="gray_r", interpolation="none")
        plt.axis("off")
        plt.title(f"Solution #{idx+1}")
        plt.show()

if __name__ == "__main__":
    row_constraints = [[1], [1], [1], [1], [1], [1], [1]]
    col_constraints = [[1], [1], [1], [1], [1], [1], [1]]
    solver = NonogramSolverWithCP(row_constraints, col_constraints)
    start_time = time.time()
    solutions = solver.solve()
    end_time = time.time()
    print(f"共找到 {len(solutions)} 个解")
    print(f"求解耗时: {end_time - start_time:.8f} 秒")
    display_solutions(solutions)