import numpy
from scipy.special import expit as sigmoid
from scipy.special import logit as logit

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes    # количество узлов входного слоя
        self.hnodes = hiddennodes   # скрытого слоя
        self.onodes = outputnodes   # выходгого слоя
        self.lr = learningrate      # коэффициент обучения, он же - шаг градиентного спуска

        # матрица весов связей входного слоя со скрытым
        self.wih = numpy.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes, self.inodes))

        # матрица весов связей скрытого слоя с выходным
        self.who = numpy.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes, self.hnodes))

        # фунуция активации (сигмоида)
        self.activation_function = lambda x: sigmoid(x)

        # обратная функция активации для обратного прохода
        self.inverse_activation_function = lambda x: logit(x)
        pass

    def train(self, inputs_list, targets_list):

        # преобразуем массивы входных и целевых значений в вектора (транспонируем)
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # прогоняем входные значения через нейросеть
        # получаем значения выходного и скрытого слоёв
        final_outputs, hidden_outnputs = self.query(inputs_list)

        # вычисляем значения ошибки на выходном слое
        output_errors = targets - final_outputs

        # и на скрытом слое путём умножения матрицы связей между скрытым и выходным слоями на вектор выходных ошибок
        hidden_errors = numpy.dot(self.who.T, output_errors)
       # с помощью метода градиентного спуска корректируем веса связей в обоих матрицах
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outnputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outnputs * (1.0 - hidden_outnputs)),
                                        numpy.transpose(inputs))
        pass

    # Опрос нейросети
    # Формирует ответ на входные данные на основании обучения
    def query(self, inputs_list):

        # массив целевых значений транспонируется и теперь представляет собой вектор
        inputs = numpy.array(inputs_list, ndmin=2).T

        # полученный вектор умножается на матрицу весов связей входного слоя со скрытым
        # получаем вектор входных значений скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # эти значения пропускаются через функцию активации
        # получаем выходные значения скрытого слоя
        hidden_outnputs = self.activation_function(hidden_inputs)
        # выходные значения скрытого слоя умножаются на матрицу весов связей скрытого слоя с выходным
        final_inputs = numpy.dot(self.who, hidden_outnputs)
        # и пропускаются через функцию активации
        final_outputs = self.activation_function(final_inputs)

        # в итоге имеем выходные значения выходного и скрытого слоёв
        return final_outputs, hidden_outnputs

    def backquery(self, targets_list):

        # транспонирует массив целевых значений в вертикальный
        final_outputs = numpy.array(targets_list, ndmin=2).T

        # расчитываем сигналы на входе выходного слоя пропуская их через обратную функцию активации
        final_inputs = self.inverse_activation_function(final_outputs)

        # расчитываем сигналы на выходе скрытого слоя умножая матрицу связей между скрытым и выходным слоями
        # на входные сигналы выходного слоя
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # масштабируем их к значениям от 0.01 до 0.99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # аналогично расчитываем сигналы на входе скрытого слоя
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # и сигналы на входном слое
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # масштабируем их к значениям от 0.01 до 0.99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        # возвращаем значения, которые моглы быть поданы на вход нейросети при таких целевых значениях,
        # которые мы только что через неё прогнали
        return inputs