import numpy as np
import matplotlib.pyplot as plt


def reward_calc(HE,CTE,r):
    R1 = - 2*abs(HE) - abs(CTE) - abs(r)
    return R1


# testing

# R1 = [-2,1]

# fig = plt.figure(figsize = (12,10))
# ax = plt.axes(projection='3d')

# he = np.arange(-1, 1, 0.01)
# cte = np.arange(-1, 1, 0.01)

# HE, CTE = np.meshgrid(he, cte)
# reward = 1 - 2*abs(HE) - abs(CTE)
# surf = ax.plot_surface(HE, CTE, reward, cmap = plt.cm.cividis)

# Z = 0*abs(HE) + 0*abs(CTE)
# surf_plane = ax.plot_surface(HE,CTE,Z)

# ax.set_xlabel('he', labelpad=20)
# ax.set_ylabel('cte', labelpad=20)
# ax.set_zlabel('rew', labelpad=20)

# fig.colorbar(surf_plane,shrink=0.5,aspect = 8)
# fig.colorbar(surf, shrink=0.5, aspect=8)


# plt.show()
# print(np.max(reward))