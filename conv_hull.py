import numpy as np
from point import *
import matplotlib.pyplot as plt 

x_min = 0
y_min = 0
x_max = 100
x_max = 100
point_set = PointSet.random_region(x_min, y_min, x_max, x_max, 10)
border_set = point_set.get_convex_hull()
border_set.print()
point_set.plot_convex_hull_demo()
plt.savefig('figures/ch_1.png', bbox_inches = 'tight')

point_set = PointSet.random_region(x_min, y_min, x_max, x_max, 30)
border_set = point_set.get_convex_hull()
border_set.print()
point_set.plot_convex_hull_demo()
plt.savefig('figures/ch_2.png', bbox_inches = 'tight')

point_set = PointSet.random_region(x_min, y_min, x_max, x_max, 100)
border_set = point_set.get_convex_hull()
border_set.print()
point_set.plot_convex_hull_demo()
plt.savefig('figures/ch_3.png', bbox_inches = 'tight')

point_set = PointSet.random_region(x_min, y_min, x_max, x_max, 300)
border_set = point_set.get_convex_hull()
border_set.print()
point_set.plot_convex_hull_demo()
plt.savefig('figures/ch_4.png', bbox_inches = 'tight')

plt.show()