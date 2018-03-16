import re
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy as sp

def arr_summ(arr):
    if(len(arr)>1):
        return arr[0]+arr_summ(arr[1:])
    else:
        return arr[0]

def numerological_minimization(num):
    nums = [int(x) for x in re.findall(r'\d', num)]

    result = arr_summ(nums)
    while len(nums)>1:
        result = arr_summ(nums)
        nums = [int(x) for x in list(str(result))]

    return result

def sum_product(x, y):
    sum_prod = 0
    for i in range(0, len(x)):
        sum_prod += x[i] * y[i]
    return sum_prod

def sum_square(x):
    sum_sqrt = 0
    for i in range(0, len(x)):
        sum_sqrt += x[i]*x[i]
    return sum_sqrt

def square_arr(x):
    return [x*x for x in x]

data = ['25.07.1996','25.07.1997','25.07.1998','25.07.1999','25.07.2000','25.07.2001','25.07.2002','25.07.2003','25.07.2004',
        '25.07.2005','25.07.2006','25.07.2007','25.07.2008','25.07.2009','25.07.2010','25.07.2011','25.07.2012','25.07.20103',
        '25.07.2014','25.07.2015','25.07.2016','25.07.2017','25.07.2018']

x = [1,	2,	3,	4,	5,	6,	7,	8,	9,	10]
y = [0.2,	0.25,	0.272727273,	0.285714286,	0.294117647,	0.3,	0.304347826,	0.307692308,	0.310344828,	0.3125]

# x, y = np.loadtxt("1880-2018.csv", delimiter=',', skiprows=5, unpack=True)

# Метод наименьших квадратов для анализа линейной зависимости

A = (sum(x)*sum(y)-len(x)*sum_product(x,y))/(sum(x)*sum(x)-len(x)*sum_square(x))
B = (sum(y) - A*sum(x))/len(x)
print(A, B)

fx = [A*x+B for x in x]
print(fx)

# Для анализа нелинейной зависимости

X1 = x[0]
X2 = x[len(x)-1]
Y1 = y[0]
Y2 = y[len(y)-1]

Xar = (X1+X2)/2
Xgeom = math.sqrt(X1*X2)
Xgarm = (2*X1*X2)/(X1+X2)

Yar = (Y1+Y2)/2
Ygeom = math.sqrt(Y1*Y2)
Ygarm = (2*Y1*Y2)/(Y1+Y2)

Yar_pr = A*Xar+B
Ygeom_pr = A*Xgeom+B
Ygarm_pr = A*Xgarm+B

# Вычислим пограшности, в зависимости от того, какая из них минимальна, выберем вид аналитической зависимости, наиболее
# точно описывающий выборку

e1 = abs(Yar_pr - Yar)      # погрешность относительно линейной зависимости
e2 = abs(Yar_pr - Ygeom)    # погрешность относительно показательной зависимости
e3 = abs(Yar_pr - Ygarm)    # погрешность относительно дробно-рациональной вида y = 1/(Ax+B)
e4 = abs(Ygeom_pr - Yar)    # погрешность относительно логарифмической
e5 = abs(Ygeom_pr - Ygeom)  # погрешность относительно смешанной
e6 = abs(Ygarm_pr - Yar)    # погрешность относительно гиперболической
e7 = abs(Ygarm_pr - Ygarm)  # погрешность относительно дробно-рациональной вида y = x/(Ax+B)

print(min([e1,e2,e3,e4,e5,e6,e7]))

# Произведём уточнение коэффициентов, в зависимости от выбранной функции

Yres = []
if min([e1,e2,e3,e4,e5,e6,e7]) == e7:
    M = np.array([[sum_square(y), sum_product(x, square_arr(y))],
                  [sum_product(x, square_arr(y)), sum_product(square_arr(x), square_arr(y))]])
    V = np.array([sum_product(x,y), sum_product(square_arr(x), y)])
    R = np.linalg.solve(M,V)
    print(R)
    Yres = [x/(R[0]*x+R[1]) for x in [1,2,3,4,5,6,7,8,9,10]]
    print(Yres)

# fig = plt.scatter(x,y, color='b')
# plt.plot(x, fx, color='r')
# plt.plot(x, Yres, color='g')
# plt.grid(True)
# plt.show()



fp, residuals, rank, sv, rcond = sp.polyfit(x,y,5, full=True)
f = sp.poly1d(fp)

fig = plt.scatter(x,y, color='b')
plt.plot(x, f(x))
plt.plot(x, fx, color='r')
plt.plot(x, Yres, color='g')
plt.grid(True)
plt.show()
