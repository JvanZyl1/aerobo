from .rover_utils import RoverDomain, PointBSpline, ConstObstacleCost, NegGeom, AABoxes, UnionGeom, AdditiveCosts, ConstCost
import numpy as np
import random
from shapely.geometry import Point, LineString, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection



class NormalizedInputFn:
    def __init__(self, fn_instance, x_range):
        self.fn_instance = fn_instance
        self.x_range = x_range

    def __call__(self, x):
        cost, points = self.fn_instance(self.project_input(x))
        return cost, points

    def project_input(self, x):
        return x * (self.x_range[1] - self.x_range[0]) + self.x_range[0]

    def inv_project_input(self, x):
        return (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0])

    def get_range(self):
        return np.array([np.zeros(self.x_range[0].shape[0]), np.ones(self.x_range[0].shape[0])])
    
    
class ConstantOffsetFn:
    def __init__(self, fn_instance, offset):
        self.fn_instance = fn_instance
        self.offset = offset

    def __call__(self, x):
        cost, points = self.fn_instance(x)
        return cost + self.offset, points

    def get_range(self):
        return self.fn_instance.get_range()

def create_cost_large():
    c = np.array([[0.43143755, 0.20876147],
                  [0.38485367, 0.39183579],
                  [0.02985961, 0.22328303],
                  [0.7803707, 0.3447003],
                  [0.93685657, 0.56297285],
                  [0.04194252, 0.23598362],
                  [0.28049582, 0.40984475],
                  [0.6756053, 0.70939481],
                  [0.01926493, 0.86972335],
                  [0.5993437, 0.63347932],
                  [0.57807619, 0.40180792],
                  [0.56824287, 0.75486851],
                  [0.35403502, 0.38591056],
                  [0.72492026, 0.59969313],
                  [0.27618746, 0.64322757],
                  [0.54029566, 0.25492943],
                  [0.30903526, 0.60166842],
                  [0.2913432, 0.29636879],
                  [0.78512072, 0.62340245],
                  [0.29592116, 0.08400595],
                  [0.87548394, 0.04877622],
                  [0.21714791, 0.9607346],
                  [0.92624074, 0.53441687],
                  [0.53639253, 0.45127928],
                  [0.99892031, 0.79537837],
                  [0.84621631, 0.41891986],
                  [0.39432819, 0.06768617],
                  [0.92365693, 0.72217512],
                  [0.95520914, 0.73956575],
                  [0.820383, 0.53880139],
                  [0.22378049, 0.9971974],
                  [0.34023233, 0.91014706],
                  [0.64960636, 0.35661133],
                  [0.29976464, 0.33578931],
                  [0.43202238, 0.11563227],
                  [0.66764947, 0.52086962],
                  [0.45431078, 0.94582745],
                  [0.12819915, 0.33555344],
                  [0.19287232, 0.8112075],
                  [0.61214791, 0.71940626],
                  [0.4522542, 0.47352186],
                  [0.95623345, 0.74174186],
                  [0.17340293, 0.89136853],
                  [0.04600255, 0.53040724],
                  [0.42493468, 0.41006649],
                  [0.37631485, 0.88033853],
                  [0.66951947, 0.29905739],
                  [0.4151516, 0.77308712],
                  [0.55762991, 0.26400156],
                  [0.6280609, 0.53201974],
                  [0.92727447, 0.61054975],
                  [0.93206587, 0.42107549],
                  [0.63885574, 0.37540613],
                  [0.15303425, 0.57377797],
                  [0.8208471, 0.16566631],
                  [0.14889043, 0.35157346],
                  [0.71724622, 0.57110725],
                  [0.32866327, 0.8929578],
                  [0.74435871, 0.47464421],
                  [0.9252026, 0.21034329],
                  [0.57039306, 0.54356078],
                  [0.56611551, 0.02531317],
                  [0.84830056, 0.01180542],
                  [0.51282028, 0.73916524],
                  [0.58795481, 0.46527371],
                  [0.83259048, 0.98598188],
                  [0.00242488, 0.83734691],
                  [0.72505789, 0.04846931],
                  [0.07312971, 0.30147979],
                  [0.55250344, 0.23891255],
                  [0.51161315, 0.46466442],
                  [0.802125, 0.93440495],
                  [0.9157825, 0.32441602],
                  [0.44927665, 0.53380074],
                  [0.67708372, 0.67527231],
                  [0.81868924, 0.88356194],
                  [0.48228814, 0.88668497],
                  [0.39805433, 0.99341196],
                  [0.86671752, 0.79016975],
                  [0.01115417, 0.6924913],
                  [0.34272199, 0.89543756],
                  [0.40721675, 0.86164495],
                  [0.26317679, 0.37334193],
                  [0.74446787, 0.84782643],
                  [0.55560143, 0.46405104],
                  [0.73567977, 0.12776233],
                  [0.28080322, 0.26036748],
                  [0.17507419, 0.95540673],
                  [0.54233783, 0.1196808],
                  [0.76670967, 0.88396285],
                  [0.61297539, 0.79057776],
                  [0.9344029, 0.86252764],
                  [0.48746839, 0.74942784],
                  [0.18657635, 0.58127321],
                  [0.10377802, 0.71463978],
                  [0.7771771, 0.01463505],
                  [0.7635042, 0.45498358],
                  [0.83345861, 0.34749363],
                  [0.38273809, 0.51890558],
                  [0.33887574, 0.82842507],
                  [0.02073685, 0.41776737],
                  [0.68754547, 0.96430979],
                  [0.4704215, 0.92717361],
                  [0.72666234, 0.63241306],
                  [0.48494401, 0.72003268],
                  [0.52601215, 0.81641253],
                  [0.71426732, 0.47077212],
                  [0.00258906, 0.30377501],
                  [0.35495269, 0.98585155],
                  [0.65507544, 0.03458909],
                  [0.10550588, 0.62032937],
                  [0.60259145, 0.87110846],
                  [0.04959159, 0.535785]])

    l = c - 0.025
    h = c + 0.025

    r_box = np.array([[0.5, 0.5]])
    r_l = r_box - 0.5
    r_h = r_box + 0.5

    trees = AABoxes(l, h)
    r_box = NegGeom(AABoxes(r_l, r_h))
    obstacles = UnionGeom([trees, r_box])

    start = np.zeros(2) + 0.05
    goal = np.array([0.95, 0.95])

    costs = [ConstObstacleCost(obstacles, cost=20.), ConstCost(0.05)]
    cost_fn = AdditiveCosts(costs)
    return cost_fn, start, goal


def create_large_domain(force_start=False,
                        force_goal=False,
                        start_miss_cost=None,
                        goal_miss_cost=None):
    
    cost_fn, start, goal = create_cost_large()

    n_points = 30
    traj = PointBSpline(dim=2, num_points=n_points)
    n_params = traj.param_size
    domain = RoverDomain(cost_fn,
                         start=start,
                         goal=goal,
                         traj=traj,
                         start_miss_cost=start_miss_cost,
                         goal_miss_cost=goal_miss_cost,
                         force_start=force_start,
                         force_goal=force_goal,
                         s_range=np.array([[-0.1, -0.1], [1.1, 1.1]]))
    return domain

class ObstacleConstraints:
    def __init__(self, domain_bounds, num_obstacles=15, seed=42):
        """
        Initialize with random fixed circular obstacles within the domain.
        """
        random.seed(seed)
        self.obstacles, self.trees = self._generate_fixed_circular_obstacles(domain_bounds, num_obstacles)

    def _generate_fixed_circular_obstacles(self, bounds, num_obstacles):
        min_x, max_x = bounds
        min_y, max_y = bounds
        obstacles = []
        X = np.array([
            0.57548412, 
            0.21752639, 
            0.46282409, 
            0.50296161, 
            0.37972964,
            0.19677418, 
            0.02388237, 
            0.58489599, 
            0.19839656, 
            0.72848741,
            0.72523733, 
            0.30622546, 
            0.86149176, 
            0.08347126, 
            0.76274493
            ])
        Y = np.array([
            0.02250968, 
            0.20088966, 
            0.60902954, 
            0.37824495, 
            0.0268175,
            0.45481976, 
            0.17895389, 
            0.49044733, 
            0.53033912, 
            0.00584888,
            0.32832546, 
            0.43993155, 
            0.30293509, 
            0.08704474, 
            0.44335343
            ])
        
        for i in range(num_obstacles):
            # Ensure the rectangle stays within bounds
            width,height = 0.1,0.1
            # (x,y) define lower-left corner of rectangle
            x,y = X[i],Y[i]

            rect = Polygon([
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height)
            ])
            obstacles.append(rect)

        c = np.array([[0.43143755, 0.20876147],
                  [0.38485367, 0.39183579],
                  [0.02985961, 0.22328303],
                  [0.7803707, 0.3447003],
                  [0.93685657, 0.56297285],
                  [0.04194252, 0.23598362],
                  [0.28049582, 0.40984475],
                  [0.6756053, 0.70939481],
                  [0.01926493, 0.86972335],
                  [0.5993437, 0.63347932],
                  [0.57807619, 0.40180792],
                  [0.56824287, 0.75486851],
                  [0.35403502, 0.38591056],
                  [0.72492026, 0.59969313],
                  [0.27618746, 0.64322757],
                  [0.54029566, 0.25492943],
                  [0.30903526, 0.60166842],
                  [0.2913432, 0.29636879],
                  [0.78512072, 0.62340245],
                  [0.29592116, 0.08400595],
                  [0.87548394, 0.04877622],
                  [0.21714791, 0.9607346],
                  [0.92624074, 0.53441687],
                  [0.53639253, 0.45127928],
                  [0.99892031, 0.79537837],
                  [0.84621631, 0.41891986],
                  [0.39432819, 0.06768617],
                  [0.92365693, 0.72217512],
                  [0.95520914, 0.73956575],
                  [0.820383, 0.53880139],
                  [0.22378049, 0.9971974],
                  [0.34023233, 0.91014706],
                  [0.64960636, 0.35661133],
                  [0.29976464, 0.33578931],
                  [0.43202238, 0.11563227],
                  [0.66764947, 0.52086962],
                  [0.45431078, 0.94582745],
                  [0.12819915, 0.33555344],
                  [0.19287232, 0.8112075],
                  [0.61214791, 0.71940626],
                  [0.4522542, 0.47352186],
                  [0.95623345, 0.74174186],
                  [0.17340293, 0.89136853],
                  [0.04600255, 0.53040724],
                  [0.42493468, 0.41006649],
                  [0.37631485, 0.88033853],
                  [0.66951947, 0.29905739],
                  [0.4151516, 0.77308712],
                  [0.55762991, 0.26400156],
                  [0.6280609, 0.53201974],
                  [0.92727447, 0.61054975],
                  [0.93206587, 0.42107549],
                  [0.63885574, 0.37540613],
                  [0.15303425, 0.57377797],
                  [0.8208471, 0.16566631],
                  [0.14889043, 0.35157346],
                  [0.71724622, 0.57110725],
                  [0.32866327, 0.8929578],
                  [0.74435871, 0.47464421],
                  [0.9252026, 0.21034329],
                  [0.57039306, 0.54356078],
                  [0.56611551, 0.02531317],
                  [0.84830056, 0.01180542],
                  [0.51282028, 0.73916524],
                  [0.58795481, 0.46527371],
                  [0.83259048, 0.98598188],
                  [0.00242488, 0.83734691],
                  [0.72505789, 0.04846931],
                  [0.07312971, 0.30147979],
                  [0.55250344, 0.23891255],
                  [0.51161315, 0.46466442],
                  [0.802125, 0.93440495],
                  [0.9157825, 0.32441602],
                  [0.44927665, 0.53380074],
                  [0.67708372, 0.67527231],
                  [0.81868924, 0.88356194],
                  [0.48228814, 0.88668497],
                  [0.39805433, 0.99341196],
                  [0.86671752, 0.79016975],
                  [0.01115417, 0.6924913],
                  [0.34272199, 0.89543756],
                  [0.40721675, 0.86164495],
                  [0.26317679, 0.37334193],
                  [0.74446787, 0.84782643],
                  [0.55560143, 0.46405104],
                  [0.73567977, 0.12776233],
                  [0.28080322, 0.26036748],
                  [0.17507419, 0.95540673],
                  [0.54233783, 0.1196808],
                  [0.76670967, 0.88396285],
                  [0.61297539, 0.79057776],
                  [0.9344029, 0.86252764],
                  [0.48746839, 0.74942784],
                  [0.18657635, 0.58127321],
                  [0.10377802, 0.71463978],
                  [0.7771771, 0.01463505],
                  [0.7635042, 0.45498358],
                  [0.83345861, 0.34749363],
                  [0.38273809, 0.51890558],
                  [0.33887574, 0.82842507],
                  [0.02073685, 0.41776737],
                  [0.68754547, 0.96430979],
                  [0.4704215, 0.92717361],
                  [0.72666234, 0.63241306],
                  [0.48494401, 0.72003268],
                  [0.52601215, 0.81641253],
                  [0.71426732, 0.47077212],
                  [0.00258906, 0.30377501],
                  [0.35495269, 0.98585155],
                  [0.65507544, 0.03458909],
                  [0.10550588, 0.62032937],
                  [0.60259145, 0.87110846],
                  [0.04959159, 0.535785]])
        
        trees = []
        for tree in c:
            # Ensure the rectangle stays within bounds
            width,height = 0.05,0.05
            # (x,y) define lower-left corner of rectangle
            x,y = tree[0],tree[1]

            rect = Polygon([
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height)
            ])
            trees.append(rect)
        
        return obstacles, trees

    def evaluate_constraints(self, trajectory_points):
        """
        Given a trajectory (Nx2 numpy array), evaluate 15 constraint values.
        """
        traj_line = LineString(trajectory_points)
        constraints = []

        for oi in self.obstacles:
            if not traj_line.intersects(oi):
                # Trajectory avoids the obstacle
                dist = oi.distance(traj_line)
                constraints.append(-dist)
            else:
                # Trajectory intersects the obstacle
                intersected_geom = traj_line.intersection(oi)
                
                # Extract intersection points
                if intersected_geom.geom_type == 'LineString':
                    intersect_points = [Point(pt) for pt in np.array(intersected_geom.coords)]
                else:
                    intersect_points = []

                boundary = oi.boundary
                max_min_dist = 0
                for alpha in intersect_points:
                    min_dist = boundary.distance(alpha)
                    max_min_dist = max(max_min_dist, min_dist)
                constraints.append(max_min_dist)

        return constraints

    def plot_obstacles(self, trajectory_points, path, ax, color='midnightblue'):
        """
        Optionally, plot the obstacles (requires matplotlib).
        """

        # Plotting stuff
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import rc
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
        matplotlib.rcParams['text.usetex'] = True

        traj_line = LineString(trajectory_points)

        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        
        for tree in self.trees:
            x, y = tree.exterior.xy
            ax.fill(x, y, color='tan', alpha=0.75, label='_nolegend_')


        for obs in self.obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, color='darkgreen', alpha=1, label='_nolegend_')

        # get points on the current trajectory
        traj_points = np.array(traj_line.xy)
        ax.plot(traj_points[0,:], traj_points[1,:], 'darkred',linewidth=3, label='_nolegend_')
        # plot start and end point
        ax.plot(traj_points[0,0], traj_points[1,0], 'o', color='rosybrown',markersize=8, label='_nolegend_')
        ax.plot(traj_points[0,-1], traj_points[1,-1], 'o', color='rosybrown',markersize=8, label='_nolegend_')

        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_facecolor("whitesmoke")

        return ax

class Rover:
    def __init__(self):
        def l2cost(x, point):
            return 10 * np.linalg.norm(x - point, 1)

        domain = create_large_domain(force_start=True,
                                     force_goal=True,
                                     start_miss_cost=l2cost,
                                     goal_miss_cost=l2cost)
        n_points = domain.traj.npoints

        raw_x_range = np.repeat(domain.s_range, n_points, axis=1)

        # maximum value of f
        f_max = 5.0
        f = NormalizedInputFn(
            ConstantOffsetFn(domain, f_max), 
            raw_x_range
            )
        x_range = f.get_range()
        
        self.f = f
        self.dims = 60
        self.lb = x_range[0]
        self.ub = x_range[1]
        self.bounds = x_range
        self.opt_val = f_max
        self.num_constraints = 15

        constraints = ObstacleConstraints(
            domain_bounds=x_range[:,0],
            num_obstacles=self.num_constraints
            )
        
        self.eval_constraints = constraints.evaluate_constraints
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        f,trajectory = self.f(x)

        # transform points on trajectory into constraints
        c = self.eval_constraints(trajectory)
        
        return f,c
    
    def plot_trajectory(self,x,path,ax):
        f,traj = self.f(x)
        #print(f'Objective vlaue: {f}')
        constraints = ObstacleConstraints(
            domain_bounds=self.bounds[:,0],
            num_obstacles=self.num_constraints
            )
        constraints.plot_obstacles(traj,path,ax)



def main():
    func = Rover()
    x = np.random.uniform(func.lb, func.ub)
    y = func(x)
    print('Input = {}'.format(x))
    print('Output = {}'.format(func(x)))


if __name__ == "__main__":
    main()