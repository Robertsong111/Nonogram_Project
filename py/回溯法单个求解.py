import time
import matplotlib.pyplot as plt


class NonogramSolver:
    def __init__(self, row_constraints, col_constraints):
        self.row_constraints = row_constraints
        self.col_constraints = col_constraints
        self.m = len(row_constraints)
        self.n = len(col_constraints)
        self.board = [[0] * self.n for _ in range(self.m)]
        self.solutions = []

    def is_valid(self):
        for r in range(self.m):
            if not self.check_line(self.board[r], self.row_constraints[r]):
                return False
        for c in range(self.n):
            if not self.check_line([self.board[r][c] for r in range(self.m)], self.col_constraints[c]):
                return False
        return True

    def check_line(self, line, constraint):
        current_block = 0
        blocks = []
        for cell in line:
            if cell == 1:
                current_block += 1
            elif current_block > 0:
                blocks.append(current_block)
                current_block = 0
        if current_block > 0:
            blocks.append(current_block)
        return blocks == constraint

    def solve(self, row=0, col=0):
        if row == self.m:
            if self.is_valid():
                self.solutions.append([r[:] for r in self.board])
            return

        if col == self.n:
            self.solve(row + 1, 0)
            return

        self.board[row][col] = 1
        self.solve(row, col + 1)

        self.board[row][col] = 0
        self.solve(row, col + 1)

    def get_solutions(self):
        return self.solutions


# ======== 测试代码 ===========
if __name__ == "__main__":
    start_time = time.time()

    row_constraints = [[1], [1], [1], [1], [1]]
    col_constraints = [[1], [1], [1], [1], [1]]

    solver = NonogramSolver(row_constraints, col_constraints)
    solver.solve()
    solutions = solver.get_solutions()
    end_time = time.time()
    print(f"程序运行时间: {end_time - start_time:.4f} 秒")
    print(f"共找到 {len(solutions)} 个解：")
    for solution in solutions:
        # imshow 黑白反转，1 表示黑格（显示为黑）
        plt.imshow([[1 - cell for cell in row] for row in solution], cmap="gray", interpolation="none")
        plt.axis('off')
        plt.show()


