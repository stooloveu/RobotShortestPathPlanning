import numpy as np
from point import *
import matplotlib.pyplot as plt 


# pw = PointWorld(start_and_end = True, startp = Point(0, 0), endp = Point(100, 100))
# pw.add_convex_polygon(10, 10, 20, 20, random_n = 5)
# pw.add_convex_polygon(20, 20, 40, 40)
# pw.add_convex_polygon(50, 50, 90, 90)
# pw.add_convex_polygon(10, 50, 60, 90)
# pw.add_convex_polygon(50, 10, 90, 60)

# pw.add_convex_polygon(20, 50, 50, 80, random_n = 10)
# # for poly in pw.point_sets:
# #     poly.print()
# # pw.show_vis_graph(show = True)
# pw.show_vis_graph(show = False)
# plt.savefig('figures/vis_graph_2.png', bbox_inches = 'tight')

# pw.show_shortest_path(show = False)
# plt.savefig('figures/shortest_path_2.png', bbox_inches = 'tight')


pw = PointWorld(start_and_end = True, startp = Point(0, 0), endp = Point(100, 100))
pw.add_convex_polygon(10, 10, 20, 20, random_n = 5)
pw.add_convex_polygon(10, 10, 20, 20, random_n = 5)
pw.add_convex_polygon(10, 70, 30, 80, random_n = 5)
pw.add_convex_polygon(50, 30, 70, 40, random_n = 5)
pw.add_convex_polygon(90, 30, 100, 40, random_n = 5)
pw.add_convex_polygon(40, 60, 50, 70, random_n = 5)
pw.add_convex_polygon(10, 70, 20, 80, random_n = 5)
pw.add_convex_polygon(60, 30, 70, 40, random_n = 5)

pw.add_convex_polygon(20, 20, 40, 40)
pw.add_convex_polygon(50, 50, 90, 90)


pw.add_convex_polygon(20, 50, 50, 80, random_n = 10)
# for poly in pw.point_sets:
#     poly.print()
# pw.show_vis_graph(show = True)
pw.show_vis_graph(show = False)
plt.savefig('figures/vis_graph_3.png', bbox_inches = 'tight')

pw.show_shortest_path(show = False)
plt.savefig('figures/shortest_path_3.png', bbox_inches = 'tight')
