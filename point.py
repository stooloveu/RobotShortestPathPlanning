import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

class Point:
    def __init__(self, x, y, label = -1):
        self.x = 1. * x
        self.y = 1. * y
        self.label = int(label)

    def innerp(self, p2):
        return self.x * p2.x + self.y * p2.y

    def crossp(self, p2):
        return self.x * p2.y - self.y * p2.x

    def vecto(self, p2):
        return Point(p2.x - self.x, p2.y - self.y)
    
    def vecfrom(self, p2):
        return Point(self.x - p2.x, self.y - p2.y)
    
    def vecadd(self, p2):
        return Point(self.x + p2.x, self.y + p2.y)

    def norm(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def angle(self):
        return np.arctan2(self.y, self.x)

class PointSet:
    def __init__(self, x, y, label = None):
        if label is None:
            label = -1 * np.ones_like(x)
        self.x = np.array(x)
        self.y = np.array(y)
        self.label = label

    @classmethod
    def random_region(cls, x1, y1, x2, y2, n, start_label = 0):
        x = np.random.random(n) * (x2 - x1) + x1
        y = np.random.random(n) * (y2 - y1) + y1
        label = np.arange(start_label, start_label + n)
        return cls(x, y, label)

    @classmethod
    def random_convex_polygon(cls, x1, y1, x2, y2, n, start_label = 0):
        x = np.random.random(n) * (x2 - x1) + x1
        y = np.random.random(n) * (y2 - y1) + y1
        poly = cls(x, y).get_convex_hull()
        label = np.arange(start_label, start_label + poly.len())
        return cls(poly.x, poly.y, label)

    @classmethod
    def from_list(cls, point_list, start_label = 0):
        x = []
        y = []
        label = np.arange(start_label, start_label + len(point_list))
        for point in point_list:
            x.append(point.x)
            y.append(point.y)
        return cls(x, y, label)
    
    def to_list(self):
        ret = []
        for i in range(self.len()):
            ret.append(self.get_point(i))
        return ret

    def append(self, point_set):
        self.x = np.append(self.x, point_set.x)
        self.y = np.append(self.y, point_set.y)
        self.label = np.append(self.label, point_set.label)
        

    def get_point(self, idx):
        return Point(self.x[idx], self.y[idx], self.label[idx])
            
    def len(self):
        return self.x.shape[0]

    def print(self):
        print("There are {} points:".format(self.len()))
        for i in range(self.len()):
            print("{}: \t({:.2f} \t{:.2f})".format(self.label[i], self.x[i], self.y[i]))

    def get_convex_hull(self):
        y_min_idx = np.argmin(self.y)
        y_min_pt = self.get_point(y_min_idx)
        self.x[[0, y_min_idx]] = self.x[[y_min_idx, 0]]
        self.y[[0, y_min_idx]] = self.y[[y_min_idx, 0]]
        self.label[[0, y_min_idx]] = self.label[[y_min_idx, 0]]

        angles = np.zeros(self.x.shape)
        for i in range(1, self.len()):
            angles[i] = self.get_point(i).vecfrom(y_min_pt).angle()
        angles[0] = -1
        order = np.argsort(angles)

        self.x = self.x[order]
        self.y = self.y[order]
        self.label = self.label[order]

        stack = []
        stack.append(self.get_point(0))
        stack.append(self.get_point(1))
        stack.append(self.get_point(2))
        for i in range(3, self.len()):
            while stack[-2].vecto(stack[-1]).crossp(stack[-2].vecto(self.get_point(i))) <= 0:
                stack.pop()
            stack.append(self.get_point(i))
        
        self.convex_hull = PointSet.from_list(stack)

        return PointSet.from_list(stack)

    def to_edge_list_circular(self, excluded_labels = set([])):
        ret = []
        for i in range(self.len()):
            i1 = i
            i2 = i+1 if i+1 < self.len() else 0
            if self.get_point(i1).label not in excluded_labels and self.get_point(i2).label not in excluded_labels:
                ret.append((self.get_point(i1), self.get_point(i2)))
        return ret

    def plot_convex_hull_demo(self, show = False):
        plt.figure()
        plt.axis('equal')
        plt.scatter(self.x, self.y)
        self.get_convex_hull()
        self.convex_hull.plot_contour_circular(color='r')
        if show:
            plt.show()

    def plot_contour_circular(self, color = 'r'):
        plt.plot(self.x, self.y, "-o", color = color)
        plt.plot([self.x[-1], self.x[0]],[self.y[-1], self.y[0]], "-o", color = color)
        # plt.savefig('../figures/q72_acc_lr_{}.png'.format(learning_rate), bbox_inches = 'tight')
    
    def to_plt_polygon(self, shift = None):
        if shift:
            ret = matplotlib.patches.Polygon(np.vstack((self.x + shift.x, self.y + shift.y)).T)
        else:
            ret = matplotlib.patches.Polygon(np.vstack((self.x, self.y)).T)
        return ret

    def contains(self, point):
        p1 = point
        p2 = point.vecadd(Point(1000, 0))
        edges = self.to_edge_list_circular()
        count = 0
        for p3, p4 in edges:
            if line_intersect(p1, p2, p3, p4):
                count += 1

        if count % 2 == 0:
            return False
        else:
            return True

# ref: https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
def line_intersect(p1, p2, p3, p4):
    p = p1
    r = p1.vecto(p2)
    q = p3
    s = p3.vecto(p4)
    rs = r.crossp(s)
    if np.isclose(rs, 0):
        if np.isclose(p.vecto(q).crossp(r), 0):
            t0 = q.vecfrom(p).innerp(r) / r.innerp(r)
            t1 = q.vecadd(s).vecfrom(p).innerp(r) / r.innerp(r)
            if (t0>=0 and t0<=1) or (t1>=0 and t1<=1):
                return True
            else:
                return False
        else:
            return False
    else:
        t = q.vecfrom(p).crossp(s) / r.crossp(s)
        u = q.vecfrom(p).crossp(r) / r.crossp(s)
        if (t>=0 and t<=1) and (u>=0 and u<=1):
            return True
        else:
            return False

class PointWorld:
    def __init__(self, start_and_end = False, startp = None, endp = None):
        self.pc = 0 # point count
        self.psc = 0 # point set count
        self.point_sets = []
        self.start_and_end = start_and_end
        self._point_list = PointSet([], [], [])
        if start_and_end:
            self.startp_label = self.pc
            self.pc += 1
            self.startp = Point(startp.x, startp.y, self.startp_label)
            self._point_list.append(self.startp)
            self.endp_label = self.pc
            self.pc += 1 
            self.endp = Point(endp.x, endp.y, self.endp_label)
            self._point_list.append(self.endp)

    def point_len(self):
        return self.pc

    def point_set_len(self):
        return self.psc

    def add_convex_polygon(self, x1, y1, x2, y2, random_n = 20):
        self.psc += 1
        poly = PointSet.random_convex_polygon(x1, y1, x2, y2, random_n, start_label = self.pc)
        self.point_sets.append(poly)
        self.pc += poly.len()
        self._point_list.append(poly)

    def add_polygon(self, poly):
        self.psc += 1
        poly = PointSet.from_list(poly.to_list(), start_label = self.pc)
        self.point_sets.append(poly)
        self.pc += poly.len()
        self._point_list.append(poly)

    def get_point(self, idx):
        return self._point_list.get_point(idx)

    def plot_line(self, p1, p2, color = 'b'):
        plt.plot([p1.x, p2.x], [p1.y, p2.y], "-o", color = color)

    def show_polygon_world(self, show = False):
        plt.figure()
        ax = plt.gca()
        plt.axis('equal')
        if self.start_and_end:
            plt.scatter(self.startp.x, self.startp.y, c = 'g')
            plt.scatter(self.endp.x, self.endp.y, c = 'g')
        patches = []
        for poly in self.point_sets:
            patches.append(poly.to_plt_polygon())
        colors = 100*np.random.rand(len(patches))
        p = matplotlib.collections.PatchCollection(patches, alpha=0.4)
        p.set_array(np.array(colors))
        ax.add_collection(p)
        if show:
            plt.show()

    def is_visbile(self, p1, p2):
        for poly in self.point_sets:
            edges = poly.to_edge_list_circular(excluded_labels = set([p1.label, p2.label]))
            for p3, p4 in edges:
                if line_intersect(p1, p2, p3, p4):
                    return False
        return True

    def is_inside_any(self, point, other_than = None):
        for poly in self.point_sets:
            if poly is not other_than and poly.contains(point):
                return True
        return False

    def find_vis_edges(self, add_edges = False):
        self.vis_graph = [[] for _ in range(self.point_len())]
        self.valid = [True for _ in range(self.point_len())]
        self.valid[self.startp_label] = not self.is_inside_any(self.startp)
        self.valid[self.endp_label] = not self.is_inside_any(self.endp)
        
        if add_edges:
            for poly in self.point_sets:
                for p1, p2 in poly.to_edge_list_circular():
                    if self.valid[p1.label] and self.valid[p2.label]:
                        self.vis_graph[p1.label].append(p2.label)
                        self.vis_graph[p2.label].append(p1.label)

        for poly in self.point_sets:
            for point in poly.to_list():
                self.valid[point.label] = not self.is_inside_any(point, other_than=poly)

        if self.start_and_end:
            p1 = self.startp
            for i in range(2, self.point_len()):
                if i == self.startp_label: 
                    continue
                p2 = self.get_point(i)
                visible = self.is_visbile(p1, p2)
                if visible:
                    self.vis_graph[p1.label].append(p2.label)
                    self.vis_graph[p2.label].append(p1.label)
            p1 = self.endp
            for i in range(self.point_len()):
                if i == self.endp_label: 
                    continue
                p2 = self.get_point(i)
                visible = self.is_visbile(p1, p2)
                if visible:
                    self.vis_graph[p1.label].append(p2.label)
                    self.vis_graph[p2.label].append(p1.label)

        for i in range(self.point_set_len() - 1):
            poly1 = self.point_sets[i]
            for p1 in poly1.to_list():

                for j in range(i + 1, self.point_set_len()):
                    poly2 = self.point_sets[j]
                    for p2 in poly2.to_list():

                        visible = self.is_visbile(p1, p2)
                        if visible and self.valid[p1.label] and self.valid[p2.label]:
                            self.vis_graph[p1.label].append(p2.label)
                            self.vis_graph[p2.label].append(p1.label)

    def show_vis_graph(self, show = False):
        self.show_polygon_world()
        self.find_vis_edges()
        for i in range(self.point_len()):
            p1 = self.get_point(i)
            for j in self.vis_graph[i]:
                p2 = self.get_point(j)
                self.plot_line(p1, p2)
        if show:
            plt.show()

    def find_shortest_path(self):
        inf = 100000
        dist = [inf for _ in range(self.point_len())]
        prev = [None for _ in range(self.point_len())]
        q = set(range(self.point_len()))
        dist[self.endp_label] = 0
        while q:
            p = min(q, key = lambda _p: dist[_p])
            if dist[p] == inf:
                break
            
            for p2 in self.vis_graph[p]:
                new_dist = dist[p] + self.get_point(p).vecto(self.get_point(p2)).norm()
                if new_dist < dist[p2]:
                    dist[p2] = new_dist
                    prev[p2] = p
            
            q.remove(p)
        
        path = []
        p = self.startp_label
        while prev[p] is not None:
            path.append(p)
            p = prev[p]
        if path:
            path.append(p)
        return path, dist[self.endp_label]
    
    def show_shortest_path(self, show = False):
        inf = 100000
        self.find_vis_edges(add_edges=True)

        path, dist = self.find_shortest_path()
        if dist == inf or not self.valid[self.startp_label] or not self.valid[self.endp_label]:
            print("No path found!")
        else:
            self.show_polygon_world()
            for i in range(len(path)-1):
                self.plot_line(self.get_point(path[i]), self.get_point(path[i+1]))
        if show:
            plt.show()
    
class PointWorldRobot(PointWorld):
    # The first point has to (0,0) (relate to the start point), the reference point has to be the point with lowest y value, the order has to ccw
    def define_robot_shape(self, point_list):
        self.robot = PointSet.from_list(point_list)


    def update_new_polygon(self):
        new_point_sets = []
        for poly in self.point_sets:
            poly_vecs = []
            poly_angles = []
            for p1, p2 in poly.to_edge_list_circular():
                poly_vecs.append(p1.vecto(p2))
                angle = p1.vecto(p2).angle()
                if angle < 0:
                    angle += 2 * np.pi
                poly_angles.append(angle) 
            poly_idx = np.argsort(np.array(poly_angles))

            robot_vecs = []
            robot_angles = []
            for p1, p2 in self.robot.to_edge_list_circular():
                robot_vecs.append(p2.vecto(p1))
                angle = p2.vecto(p1).angle()
                if angle < 0:
                    angle += 2 * np.pi
                robot_angles.append(angle)
            robot_idx = np.argsort(np.array(robot_angles))

            new_points = [poly.get_point(0)]
            ipoly = 0
            irobot = 0
            for i in range(poly.len() + self.robot.len() - 1):
                if ipoly >= poly.len() or (irobot < self.robot.len() and poly_angles[poly_idx[ipoly]] > robot_angles[robot_idx[irobot]]):
                    new_points.append(new_points[-1].vecadd(robot_vecs[robot_idx[irobot]]))
                    irobot += 1
                else:
                    new_points.append(new_points[-1].vecadd(poly_vecs[poly_idx[ipoly]]))
                    ipoly += 1
            new_point_sets.append(new_points)
            
        self._point_list = PointSet.from_list(self._point_list.to_list()[0:2])
        self.psc = 0
        self.pc = 2
        self.point_sets = []
        for new_points in new_point_sets:
            self.add_polygon(PointSet.from_list(new_points))

    def show_shortest_path(self, show = False):
        inf = 100000
        self.find_vis_edges(add_edges=True)

        path, dist = self.find_shortest_path()
        if dist == inf or not self.valid[self.startp_label] or not self.valid[self.endp_label]:
            print("No path found!")
        else:
            self.show_polygon_world()
            for i in range(len(path)-1):
                self.plot_line(self.get_point(path[i]), self.get_point(path[i+1]))
            patches = []
            for i in range(len(path)):
                patches.append(self.robot.to_plt_polygon(shift = self.get_point(path[i])))

            ax = plt.gca()
            colors = 0*np.ones(len(patches))
            p = matplotlib.collections.PatchCollection(patches, alpha=1)
            p.set_array(np.array(colors))
            ax.add_collection(p)
        if show:
            plt.show()

    
    def show_shortest_path_only(self, show = False):
        inf = 100000
        self.find_vis_edges(add_edges=True)

        path, dist = self.find_shortest_path()
        if dist == inf or not self.valid[self.startp_label] or not self.valid[self.endp_label]:
            print("No path found!")
        else:
            for i in range(len(path)-1):
                self.plot_line(self.get_point(path[i]), self.get_point(path[i+1]))
            patches = []
            for i in range(len(path)):
                patches.append(self.robot.to_plt_polygon(shift = self.get_point(path[i])))

            ax = plt.gca()
            colors = 0*np.ones(len(patches))
            p = matplotlib.collections.PatchCollection(patches, alpha=1)
            p.set_array(np.array(colors))
            ax.add_collection(p)
        if show:
            plt.show()

def main():
    # print(line_intersect(Point(0,0), Point(0,2.5), Point(0,2.6), Point(0,3)))
    # print(line_intersect(Point(0,0), Point(1,1), Point(0,-2), Point(1,0)))

    # pw = PointWorld()
    # pw.add_convex_polygon(0, 0, 20, 20)
    # pw.add_convex_polygon(20, 20, 40, 40)
    # for poly in pw.point_sets:
    #     poly.print()
    # pw.show_vis_graph(show = True)

    pw = PointWorld(start_and_end = True, startp = Point(0, 0), endp = Point(100, 100))
    pw.add_convex_polygon(10, 10, 20, 20, random_n = 5)
    pw.add_convex_polygon(20, 20, 40, 40)
    pw.add_convex_polygon(50, 50, 90, 90)
    pw.add_convex_polygon(20, 50, 50, 80, random_n = 10)
    # for poly in pw.point_sets:
    #     poly.print()
    pw.show_vis_graph(show = True)
    pw.show_shortest_path(show = True)

if __name__== "__main__":
    main()