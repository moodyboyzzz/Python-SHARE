"""Модуль базовых алгоритмов линейной алгебры.
Задание состоит в том, чтобы имплементировать класс Matrix
(следует воспользоваться кодом из задания seminar06_1), учтя рекомендации pylint.
Для проверки кода следует использовать команду pylint matrix.py.
Pylint должен показывать 10 баллов.
Рекомендуемая версия pylint - 2.15.5
Кроме того, следует добавить поддержку исключений в отмеченных местах.
Для проверки корректности алгоритмов следует сравнить результаты с соответствующими функциями numpy.
"""
import random
import numpy as np

class Matrix:
    """
    Класс Matrix представляет собой матрицу и поддерживает различные операции
    линейной алгебры, такие как сложение, умножение, транспонирование и другие.
    """
    def __init__(self, nrows, ncols, init="zeros"):
        """Конструктор класса Matrix.
        Создаёт матрицу резмера nrows x ncols и инициализирует её методом init.
        nrows - количество строк матрицы
        ncols - количество столбцов матрицы
        init - метод инициализации элементов матрицы:
            "zeros" - инициализация нулями
            "ones" - инициализация единицами
            "random" - случайная инициализация
            "eye" - матрица с единицами на главной диагонали
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("nrows and ncols must be non-negative")
        if init not in ["zeros", "ones", "random", "eye"]:
            raise ValueError("Invalid initialization method")
        self.nrows = nrows
        self.ncols = ncols
        self.data = []
        if init == "zeros":
            self.data = [[0 for _ in range(self.ncols)] for _ in range(self.nrows)]
        elif init == "ones":
            self.data = [[1 for _ in range(self.ncols)] for _ in range(self.nrows)]
        elif init == "random":
            self.data = [[random.random() for _ in range(self.ncols)] for _ in range(self.nrows)]
        elif init == "eye":
            self.data = [[1 if i == j else 0 for j in range(self.ncols)] for i in range(self.nrows)]

    @staticmethod
    def from_dict(data):
        "Десериализация матрицы из словаря"
        ncols = data["ncols"]
        nrows = data["nrows"]
        items = data["data"]
        assert len(items) == ncols*nrows
        result = Matrix(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                result[(row, col)] = items[ncols*row + col]
        return result

    @staticmethod
    def to_dict(matr):
        "Сериализация матрицы в словарь"
        assert isinstance(matr, Matrix)
        nrows, ncols = matr.shape()
        data = []
        for row in range(nrows):
            for col in range(ncols):
                data.append(matr[(row, col)])
        return {"nrows": nrows, "ncols": ncols, "data": data}

    def __str__(self):
        res = ""
        for row in self.data:
            res += " ".join(str(element) for element in row) + "\n"
        return res

    def __repr__(self):
        return f"Matrix({self.nrows}, {self.ncols}, 'custom')"

    def shape(self):
        "Вернуть кортеж размера матрицы (nrows, ncols)"
        return (self.nrows, self.ncols)

    def __getitem__(self, index):
        """Получить элемент матрицы по индексу index
        index - список или кортеж, содержащий два элемента
        """
        if not isinstance(index, (tuple, list)) or len(index) != 2:
            raise ValueError("Index must be a list or tuple with two elements")
        row, col = index
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            raise IndexError("Index out of bounds")
        return self.data[row][col]

    def __setitem__(self, index, value):
        """Задать элемент матрицы по индексу index
        index - список или кортеж, содержащий два элемента
        value - Устанавливаемое значение
        """
        if not isinstance(index, (tuple, list)) or len(index) != 2:
            raise ValueError("Index must be a list or tuple with two elements")
        row, col = index
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            raise IndexError("Index out of bounds")
        self.data[row][col] = value

    def __sub__(self, rhs):
        "Вычесть матрицу rhs и вернуть результат"
        if self.nrows != rhs.nrows or self.ncols != rhs.ncols:
            raise ValueError("Matrix sizes do not match for subtraction")
        result = Matrix(self.nrows, self.ncols)
        for row in range(self.nrows):
            for col in range(self.ncols):
                result[(row, col)] = self[(row, col)] - rhs[(row, col)]
        return result

    def __add__(self, rhs):
        "Сложить с матрицей rhs и вернуть результат"
        if self.nrows != rhs.nrows or self.ncols != rhs.ncols:
            raise ValueError("Matrix sizes do not match for addition")
        result = Matrix(self.nrows, self.ncols)
        for row in range(self.nrows):
            for col in range(self.ncols):
                result[(row, col)] = self[(row, col)] + rhs[(row, col)]
        return result

    def __mul__(self, rhs):
        "Умножить на матрицу rhs и вернуть результат"
        if self.ncols != rhs.nrows:
            raise ValueError("Matrix sizes do not match for multiplication")
        result = Matrix(self.nrows, rhs.ncols)
        for i in range(self.nrows):
            for j in range(rhs.ncols):
                dot_product = 0
                for k in range(self.ncols):
                    dot_product += self[(i, k)] * rhs[(k, j)]
                result[(i, j)] = dot_product
        return result

    def __pow__(self, power):
        "Возвести все элементы в степень power и вернуть результат"
        result = Matrix(self.nrows, self.ncols)
        for row in range(self.nrows):
            for col in range(self.ncols):
                result[(row, col)] = self[(row, col)] ** power
        return result

    def sum(self):
        "Вернуть сумму всех элементов матрицы"
        total = 0
        for row in range(self.nrows):
            for col in range(self.ncols):
                total += self[(row, col)]
        return total

    def det(self):
        "Вычислить определитель матрицы"
        if self.nrows != self.ncols:
            raise ValueError("Matrix must be square to compute the determinant")
        det_value = 1.0
        mat = [row[:] for row in self.data]
        for col in range(self.ncols):
            pivot_row = col
            while pivot_row < self.nrows and mat[pivot_row][col] == 0:
                pivot_row += 1
            if pivot_row == self.nrows:
                return 0.0
            if pivot_row != col:
                mat[pivot_row], mat[col] = mat[col], mat[pivot_row]
                det_value *= -1
            pivot_element = mat[col][col]
            det_value *= pivot_element
            for j in range(col, self.ncols):
                mat[col][j] /= pivot_element
            for i in range(col + 1, self.nrows):
                factor = mat[i][col]
                for j in range(col, self.ncols):
                    mat[i][j] -= factor * mat[col][j]
        for i in range(self.nrows):
            det_value *= mat[i][i]
        return det_value

    def transpose(self):
        "Транспонировать матрицу и вернуть результат"
        result = Matrix(self.ncols, self.nrows)
        for row in range(self.nrows):
            for col in range(self.ncols):
                result[(col, row)] = self[(row, col)]
        return result

    def submatrix(self, row, col):
        "Создание подматрицы без строки row и столбца col"
        submat = Matrix(self.nrows - 1, self.ncols - 1, init="zeros")
        i_offset = 0
        for i in range(self.nrows):
            if i == row:
                i_offset = 1
                continue
            j_offset = 0
            for j in range(self.ncols):
                if j == col:
                    j_offset = 1
                    continue
                submat[(i - i_offset, j - j_offset)] = self[(i, j)]
        return submat

    def inv(self):
        "Вычислить обратную матрицу и вернуть результат"
        if self.nrows != self.ncols:
            raise ArithmeticError("Matrix must be square to compute its inverse")
        det = self.det()
        if det == 0:
            raise ArithmeticError("Matrix is singular; its inverse does not exist")
        minors = Matrix(self.nrows, self.ncols, init="zeros")
        for i in range(self.nrows):
            for j in range(self.ncols):
                submat = self.submatrix(i, j)
                minors[(i, j)] = submat.det() * ((-1) ** (i + j))
        minors = minors.transpose()
        inv_matrix = Matrix(minors.nrows, minors.ncols)
        for row in range(minors.nrows):
            for col in range(minors.ncols):
                inv_matrix[(row, col)] = minors[(row, col)] * (1 / det)
        return inv_matrix

    def tonumpy(self):
        "Приведение к массиву numpy"
        numpy_array = np.zeros((self.nrows, self.ncols))
        for row in range(self.nrows):
            for col in range(self.ncols):
                numpy_array[row, col] = self[(row, col)]
        return numpy_array

def test():
    "Тесты для сравнения алгоритмов класса Matrix с функциями numpy."
    matrix_a = Matrix(2, 2, init="ones")
    matrix_b = Matrix(2, 2, init="ones")
    matrix_c = matrix_a * matrix_b
    numpy_result = np.dot(matrix_a.tonumpy(), matrix_b.tonumpy())
    assert np.allclose(matrix_c.tonumpy(), numpy_result)
    matrix_a = Matrix(2, 2, init="random")
    matrix_a_inv = matrix_a.inv()
    numpy_result = np.linalg.inv(matrix_a.tonumpy())
    assert np.allclose(matrix_a_inv.tonumpy(), numpy_result)
    matrix_a = Matrix(2, 3, init="ones")
    result_matrix = matrix_a.transpose()
    np_matrix_a = np.array([[1, 1, 1], [1, 1, 1]])
    np_result_matrix = np.transpose(np_matrix_a)
    assert np.allclose(result_matrix.tonumpy(), np_result_matrix)
    matrix_a = Matrix(3, 3, init="random")
    result_det = matrix_a.det()
    np_matrix_a = matrix_a.tonumpy()
    np_result_det = np.linalg.det(np_matrix_a)
    assert abs(result_det - np_result_det) < 1e-6

if __name__ == "__main__":
    test()
