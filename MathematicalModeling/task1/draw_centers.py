"""
# @Time    :  2020/9/13
# @Author  :  Jimou Chen
"""
import matplotlib.pyplot as plt
import numpy as np

center = np.array([[-0.10124706, 0.25642006, - 0.09359024],
                   [-0.23356946, - 0.30465014, 10.68487533],
                   [7.34213732, - 0.26447147, - 0.09359024],
                   [0.03852011, - 3.89985084, - 0.09359024]])
colors = ['r', 'b', 'y', 'g']
new_x_data = center[:, 0]
new_y_data = center[:, 1]
new_z_data = center[:, 2]
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(new_x_data, new_y_data, new_z_data, c=colors, s=20)
plt.show()
