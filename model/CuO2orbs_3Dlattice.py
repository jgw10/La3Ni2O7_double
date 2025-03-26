import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['text.usetex'] = True


class Arrow3D:
    def __init__(self, ax, x, y, z, dx, dy, dz, color='black', arrow_length_ratio=0.1):
        self.ax = ax
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.color = color
        self.arrow_length_ratio = arrow_length_ratio

    def draw(self):
        self.ax.quiver(self.x, self.y, self.z, self.dx, self.dy, self.dz, \
                       color=self.color, arrow_length_ratio=self.arrow_length_ratio, linewidth=1)


###############################################################################################
# Cu_orbs = ('d3z2r2','dx2y2','dxy','dyz','dxz')
Cu_orbs = ['d3z2r2', 'dx2y2']
# O_orbs = ('px','py','pz')
Ox_orbs = ['px']
Oy_orbs = ['py']

square_range = 1
Ox_positions = []
Oy_positions = []
for i in range(-square_range, square_range + 1):
    for j in range(-square_range, square_range + 1):
        if i % 2 == 1 and j % 2 == 0:
            Ox_positions.append([i, j, 1])
            Ox_positions.append([i, j, -1])
        elif i % 2 == 0 and j % 2 == 1:
            Oy_positions.append([i, j, 1])
            Oy_positions.append([i, j, -1])
Cu_positions = [[0, 0, 0], [2, 2, 0]]
Ox_positions = [[1, 0, 0], [-1, 0, 0], [3, 2, 0], [1, 2, 0]]
Oy_positions = [[0, 1, 0], [0, -1, 0], [2, 3, 0], [2, 1, 0]]


# 定义极坐标方程
def px_equation(theta, phi):
    return np.sqrt(3 / np.pi) / 2 * np.sin(theta) * np.cos(phi)


def py_equation(theta, phi):
    return np.sqrt(3 / np.pi) / 2 * np.sin(theta) * np.sin(phi)


def pz_equation(theta, phi):
    return np.sqrt(3 / np.pi) / 2 * np.cos(theta)


def d3z2r2_equation(theta, phi):
    return np.sqrt(5 / np.pi) / 4 * (3 * np.cos(theta) ** 2 - 1)


def dx2y2_equation(theta, phi):
    return np.sqrt(15 / np.pi) / 4 * (np.sin(theta) ** 2 * np.cos(2 * phi))


def dxy_equation(theta, phi):
    return np.sqrt(15 / np.pi) / 4 * (np.sin(theta) ** 2 * np.sin(2 * phi))


def dyz_equation(theta, phi):
    return np.sqrt(15 / np.pi) / 2 * (np.sin(theta) * np.cos(theta) * np.sin(phi))


def dxz_equation(theta, phi):
    return np.sqrt(15 / np.pi) / 2 * (np.sin(theta) * np.cos(theta) * np.cos(phi))


# 定义theta和phi的范围
theta = np.linspace(0, np.pi, 300)
phi = np.linspace(0, 2 * np.pi, 300)

# 创建网格
Theta, Phi = np.meshgrid(theta, phi)

# 创建3D图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 计算直角坐标系下的坐标
for Cu_pos in Cu_positions:
    x = Cu_pos[0]
    y = Cu_pos[1]
    z = Cu_pos[2]
    for orb in Cu_orbs:
        if orb == 'd3z2r2':
            level = 1
            a = 1.0
        elif orb == 'dx2y2':
            level = 2
            a = 0.4
        else:
            level = 1
            a = 1.0
        val = eval(orb + '_equation(Theta,Phi)')
        r = val ** 2
        colors = plt.cm.coolwarm((val - val.min()) / (val.max() - val.min()))
        X = x + r * np.sin(Theta) * np.cos(Phi)
        Y = y + r * np.sin(Theta) * np.sin(Phi)
        Z = z + r * np.cos(Theta)

        # 绘制曲面图
        ax.plot_surface(X, Y, Z, facecolors=colors, shade=True, alpha=a, zorder=level, antialiased=True, linewidth=0)
for Ox_pos in Ox_positions:
    x = Ox_pos[0]
    y = Ox_pos[1]
    z = Ox_pos[2]
    for orb in Ox_orbs:
        if orb == 'py':
            level = 1
        elif orb == 'pz':
            level = 1
        else:
            level = 1
        val = eval(orb + '_equation(Theta,Phi)')
        r = val ** 2
        colors = plt.cm.coolwarm((val - val.min()) / (val.max() - val.min()))
        X = x + r * np.sin(Theta) * np.cos(Phi)
        Y = y + r * np.sin(Theta) * np.sin(Phi)
        Z = z + r * np.cos(Theta)

        # 绘制曲面图
        ax.plot_surface(X, Y, Z, facecolors=colors, shade=True, alpha=1, zorder=level, antialiased=True, linewidth=0)
for Oy_pos in Oy_positions:
    x = Oy_pos[0]
    y = Oy_pos[1]
    z = Oy_pos[2]
    for orb in Oy_orbs:
        if orb == 'py':
            level = 1
        elif orb == 'pz':
            level = 1
        else:
            level = 1
        val = eval(orb + '_equation(Theta,Phi)')
        r = val ** 2
        colors = plt.cm.coolwarm((val - val.min()) / (val.max() - val.min()))
        X = x + r * np.sin(Theta) * np.cos(Phi)
        Y = y + r * np.sin(Theta) * np.sin(Phi)
        Z = z + r * np.cos(Theta)

        ax.plot_surface(X, Y, Z, facecolors=colors, shade=True, alpha=1, zorder=level, antialiased=True, linewidth=0)

# val = pz_equation(Theta,Phi)
# r = val**2
# colors = plt.cm.coolwarm((val - val.min()) / (val.max() - val.min()))
# X = r * np.sin(Theta) * np.cos(Phi)
# Y = r * np.sin(Theta) * np.sin(Phi)
# Z = r * np.cos(Theta)
# ax.plot_surface(X, Y, Z, facecolors=colors ,shade=True,alpha=1,antialiased=True,linewidth=0)
#########################################################################
zs = [0]
directions = [1, -1]
a1 = 15 / np.pi / 16
b1 = 1 - 3 / np.pi / 4
Cu2x = Cu_positions[1][0]
Cu2y = Cu_positions[1][1]
for z in zs:
    for drt in directions:
        ax.plot([drt * a1, drt * b1], [0, 0], [z, z], color='black')
        ax.plot([0, 0], [drt * a1, drt * b1], [z, z], color='black')
        ax.plot([drt * a1 + Cu2x, drt * b1 + Cu2x], [Cu2y, Cu2y], [z, z], color='black')
        ax.plot([Cu2x, Cu2x], [drt * a1 + Cu2y, drt * b1 + Cu2y], [z, z], color='black')

a2 = 3 / np.pi / 4
b2 = 1 - 5 / np.pi / 4
# ax.plot([0,0], [0,0], [a2,b2], color='blue',zorder = 0)
# ax.plot([0,0], [0,0], [-a2,-b2], color='blue',zorder = 0)

# ax.text(1.3, 0, 1, "$p_x$", color='black', fontsize=15,weight='bold')
# ax.text(0, 1.35, 1, "$p_y$", color='black', fontsize=15,weight='bold')
# ax.text(0.1, 0.1, 0, "$p_z$", color='black', fontsize=15,weight='bold')
# ax.text(-0.1, -0.1, 1.5, "$d_{3z^2-r^2}$", color='black', fontsize=15,weight='bold')
# ax.text(0.2, 0.2, 1, "$d_{x^2-y^2}$", color='black', fontsize=15,weight='bold')
# ax.text(0.05, 0.05, 0.35, "$t_{dO}$", color='blue', fontsize=15,weight='bold')
# ax.text(0.45, 0.2, -1, "$t_{pd}$", color='blue', fontsize=15,weight='bold')
# ax.text(-0.8, -0.8, -1, "$t_{pp}$", color='blue', fontsize=15,weight='bold')
# ax.plot([-1,-0.2], [-0.2,-1], [-1,-1], color='black')
# ax.plot([-1,-0.2], [0,0], [-0.8,0], color='blue')
# ax.text(-0.5, 0, -0.5, "$t_{pO}$", color='blue', fontsize=15,weight='bold')


# axis_point =[1,-1,-1]
# arrowx = Arrow3D(ax, axis_point[0], axis_point[1], axis_point[2], 0.5, 0, 0, color='black')
# arrowx.draw()
# arrowy = Arrow3D(ax, axis_point[0], axis_point[1], axis_point[2], 0, 0.5, 0, color='black')
# arrowy.draw()
# arrowz = Arrow3D(ax, axis_point[0], axis_point[1], axis_point[2], 0, 0, 0.5, color='black')
# arrowz.draw()
# ax.text(axis_point[0]+0.6, axis_point[1], axis_point[2], "$x$", color='black', fontsize=15,weight='bold')
# ax.text(axis_point[0], axis_point[1]+0.6, axis_point[2], "$y$", color='black', fontsize=15,weight='bold')
# ax.text(axis_point[0], axis_point[1], axis_point[2]+0.6, "$z$", color='black', fontsize=15,weight='bold')
# 设置图形参数
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置坐标轴范围
ax.set_xlim([-1.5, 3.5])
ax.set_ylim([0.8, 1.2])
# ax.set_zlim([0.6, 0.6])

ax.axis('off')
ax.set_aspect('equal')
# ax.view_init(elev=30, azim=225)

# 显示图形
plt.show()
plt.tight_layout()
# plt.savefig('CuO2bilayer_3D.png',dpi=1000)
plt.savefig('CuO2bilayer_3D.pdf')
