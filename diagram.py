import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Đọc dữ liệu
data = pd.read_csv('C:\\Users\\tiend\\Downloads\\output.csv', names=['x1', 'x2', 'model_output', 'target', 'true_function'])

# Chuyển true_function về float, ép lỗi thành NaN
data['true_function'] = pd.to_numeric(data['true_function'], errors='coerce')
data['model_output'] = pd.to_numeric(data['model_output'], errors='coerce')

# Bỏ các dòng NaN
data = data.dropna(subset=['true_function', 'model_output'])

# Chuẩn bị dữ liệu
x1 = data['x1'].values
x2 = data['x2'].values
model_output = data['model_output'].values
true_function = data['true_function'].values

# Vì dữ liệu lộn xộn, cần nội suy thành lưới
from scipy.interpolate import griddata

xi = np.linspace(min(x1), max(x1), 100)
yi = np.linspace(min(x2), max(x2), 100)
xi, yi = np.meshgrid(xi, yi)

zi_model = griddata((x1, x2), model_output, (xi, yi), method='cubic')
zi_true = griddata((x1, x2), true_function, (xi, yi), method='cubic')

# Vẽ 2 đồ thị 3D
fig = plt.figure(figsize=(14,6))

# Neural Network Output
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(xi, yi, zi_model, cmap='inferno')
ax1.set_title('Neural Network Output')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('Output')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# True Function Output
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(xi, yi, zi_true, cmap='viridis')
ax2.set_title('True Function Output')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('True Value')
fig.colorbar(surf2, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.show()
