import time

from matplotlib import pyplot as plt


class NonogramSolver:
    def __init__(self, row_constraints, col_constraints, m, n):
        self.row_constraints = row_constraints  # 行约束
        self.col_constraints = col_constraints  # 列约束
        self.m = m  # 行数
        self.n = n  # 列数
        self.board = [[0] * n for _ in range(m)]  # 初始空白棋盘
        self.solutions = []  # 存储所有解

    def is_valid(self):
        # 检查当前棋盘是否符合约束条件
        for r in range(self.m):
            if not self.check_line(self.board[r], self.row_constraints[r]):
                return False
        for c in range(self.n):
            if not self.check_line([self.board[r][c] for r in range(self.m)], self.col_constraints[c]):
                return False
        return True

    def check_line(self, line, constraint):
        # 检查一行或一列是否符合约束条件
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
        # 递归回溯
        if row == self.m:  # 已填充完所有行
            if self.is_valid():
                self.solutions.append([row[:] for row in self.board])  # 复制当前解
            return

        if col == self.n:  # 填充到当前行的最后一列，继续下一行
            self.solve(row + 1, 0)
            return

        # 尝试填充当前格子为 1 (黑色格)
        self.board[row][col] = 1
        self.solve(row, col + 1)  # 尝试下一个格子

        # 尝试填充当前格子为 0 (白色格)
        self.board[row][col] = 0
        self.solve(row, col + 1)  # 尝试下一个格子

    def get_solutions(self):
        # 返回所有的解
        return self.solutions


# 测试代码
start_time = time.time()
row_constraints = [[2], [1], [1], [1], [1]]
col_constraints = [[2], [1], [1], [1], [1]]
solver = NonogramSolver(row_constraints, col_constraints, 5, 5)
solver.solve()
solutions = solver.get_solutions()

print(f"找到 {len(solutions)} 个解:")
for solution in solutions:
    # 在显示之前反转 1 和 0
    plt.imshow(solution, cmap="binary", interpolation="none")  # "binary" colormap 直接符合 Nonogram 逻辑
    plt.axis('off')
    plt.show()
# 记录程序结束时间
end_time = time.time()

# 计算并打印程序的运行时间
print(f"程序运行时间: {end_time - start_time} 秒")
