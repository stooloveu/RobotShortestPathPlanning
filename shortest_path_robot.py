import numpy as np
from point import *
import matplotlib.pyplot as plt 


pw = PointWorldRobot(start_and_end = True, startp = Point(0, 5), endp = Point(100, 100))
# The first point has to (0,0) (relate to the start point), the reference point has to be the point with lowest y value, the order has to ccw
pw.define_robot_shape([Point(0, 0), Point(0, -5), Point(10, -5)])
pw.add_polygon(PointSet.from_list([Point(20, 20), Point(80, 20), Point(80, 80), Point(20, 80)]))


pw.show_polygon_world()
pw.update_new_polygon()


pw.show_shortest_path_only(show = False)
plt.savefig('figures/shortest_path_robot_1.png', bbox_inches = 'tight')



pw = PointWorldRobot(start_and_end = True, startp = Point(0, 5), endp = Point(100, 100))
# The first point has to (0,0) (relate to the start point), the reference point has to be the point with lowest y value, the order has to ccw
pw.define_robot_shape([Point(0, 0), Point(0, -5), Point(10, -5)])
pw.add_convex_polygon(10, 10, 20, 20, random_n = 5)
pw.add_convex_polygon(20, 20, 40, 40)
pw.add_convex_polygon(50, 30, 90, 70)


pw.show_polygon_world()
pw.update_new_polygon()


pw.show_shortest_path_only(show = False)
plt.savefig('figures/shortest_path_robot_2.png', bbox_inches = 'tight')




pw = PointWorldRobot(start_and_end = True, startp = Point(0, 5), endp = Point(100, 100))
# The first point has to (0,0) (relate to the start point), the reference point has to be the point with lowest y value, the order has to ccw
pw.define_robot_shape([Point(0, 0), Point(0, -5), Point(10, -5)])
pw.add_convex_polygon(10, 10, 20, 20, random_n = 5)
pw.add_convex_polygon(10, 10, 20, 20, random_n = 5)
pw.add_convex_polygon(10, 70, 30, 80, random_n = 5)
pw.add_convex_polygon(50, 30, 70, 40, random_n = 5)
pw.add_convex_polygon(90, 30, 100, 40, random_n = 5)
pw.add_convex_polygon(40, 60, 50, 70, random_n = 5)
pw.add_convex_polygon(10, 70, 20, 80, random_n = 5)
pw.add_convex_polygon(60, 30, 70, 40, random_n = 5)

pw.add_convex_polygon(20, 20, 40, 40)
pw.add_convex_polygon(50, 30, 90, 60)


pw.show_polygon_world()
pw.update_new_polygon()


pw.show_shortest_path_only(show = False)
plt.savefig('figures/shortest_path_robot_3.png', bbox_inches = 'tight')



pw = PointWorldRobot(start_and_end = True, startp = Point(0, 5), endp = Point(100, 100))
# The first point has to (0,0) (relate to the start point), the reference point has to be the point with lowest y value, the order has to ccw
pw.define_robot_shape([Point(0, 0), Point(0, -5), Point(10, -5)])
pw.add_convex_polygon(10, 10, 20, 50, random_n = 5)
pw.add_convex_polygon(10, 10, 40, 20, random_n = 5)
pw.add_convex_polygon(10, 70, 30, 80, random_n = 5)
pw.add_convex_polygon(50, 30, 90, 40, random_n = 5)
pw.add_convex_polygon(90, 30, 100, 40, random_n = 5)
pw.add_convex_polygon(40, 60, 80, 70, random_n = 5)
pw.add_convex_polygon(10, 70, 50, 80, random_n = 5)
pw.add_convex_polygon(60, 30, 70, 40, random_n = 5)

pw.add_convex_polygon(20, 20, 40, 40)
pw.add_convex_polygon(50, 30, 90, 60)


pw.show_polygon_world()
pw.update_new_polygon()


pw.show_shortest_path_only(show = False)
plt.savefig('figures/shortest_path_robot_4.png', bbox_inches = 'tight')