import numpy as np


def calculate(a, b):
    # 得到具体公式
    g = -0.168 * a - 0.06 * b
    L = 1 / (1 + pow(np.e, -g))
    return L


def pred():
    l = [51, 172, 43, 67, 193, 86, 198, 205, 97, 33]
    l.sort()
    print(l)


#
# index = [33, 43, 51, 67, 86, 97, 172, 193, 205]
# lay = []
# for i in range(0, 302):
#     lay.append(0)
#
# for i in range(len(lay)):
#     if i in index:
#         lay[i] = 1
#
# print(lay)
# print(sum(lay))

# import matplotlib.pyplot as plt
#
# center = np.array([[-0.10124706, 0.25642006, - 0.09359024],
#                    [-0.23356946, - 0.30465014, 10.68487533],
#                    [7.34213732, - 0.26447147, - 0.09359024],
#                    [0.03852011, - 3.89985084, - 0.09359024]])
# colors = ['r', 'b', 'y', 'g']
# new_x_data = center[:, 0]
# new_y_data = center[:, 1]
# new_z_data = center[:, 2]
# ax = plt.figure().add_subplot(111, projection='3d')
# ax.scatter(new_x_data, new_y_data, new_z_data, c=colors, s=20)
# plt.show()
