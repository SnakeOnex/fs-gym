import numpy as np
from enum import IntEnum
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class ConeClasses(IntEnum):
    YELLOW = 0
    BLUE = 1
    ORANGE = 2
    BIG = 3

def global_to_local(points, pose) -> np.ndarray:
    pos = pose[:2]; ori = pose[2]
    R = np.array([[np.cos(ori), -np.sin(ori)],
                  [np.sin(ori), np.cos(ori)]])
    points -= pos
    points = (R.T @ points.T).T
    return points

class StateRenderer:
    def __init__(self):
        self.fig, (self.global_ax, self.local_ax) = plt.subplots(1, 2)

        # styling
        self.fig.set_facecolor((0.6,0.6,0.55))
        self.local_ax.set_facecolor('grey')
        self.global_ax.set_facecolor('grey')
        self.local_ax.grid(alpha=0.6, which='major', color = (0.6,0.6,0.55))
        self.local_ax.grid(alpha=0.3, which='minor', color = (0.6,0.6,0.55))
        self.global_ax.grid(alpha=0.6, which='major', color = (0.6,0.6,0.55))
        self.global_ax.grid(alpha=0.3, which='minor', color = (0.6,0.6,0.55))
        
        # local ax setup
        self.local_ax.set_xlim(0., 24)
        self.local_ax.set_ylim(-12, 12)
        self.local_ax.set_xticks(np.arange(0,24, 1), minor = True)
        self.local_ax.set_yticks(np.arange(-12,12, 1), minor = True)
        self.local_ax.set_aspect('equal', adjustable='box')

        # global lines
        self.map_yc_line, = self.global_ax.plot([], [], '.', color='gold')
        self.map_bc_line, = self.global_ax.plot([], [], '.', color='blue')
        self.map_oc_line, = self.global_ax.plot([], [], '.', color='orange')
        self.map_big_line, = self.global_ax.plot([], [], '.', color='red')
        self.map_path_line, = self.global_ax.plot([], [], '-', color='red')
        self.map_static_lines = [self.map_yc_line, self.map_bc_line, self.map_oc_line, self.map_big_line, self.map_path_line]

        self.glob_text = self.global_ax.text(0., 0., "", color=(0,0,0), verticalalignment='top')
        self.map_car_pose, = self.global_ax.plot([], [], 'o', color=(1,0.4,0), markersize = 7)
        self.map_dynamic_lines = [self.map_car_pose]

        # local lines
        self.loc_yc_line, = self.local_ax.plot([], [], '.', color='gold')
        self.loc_bc_line, = self.local_ax.plot([], [], '.', color='blue')
        self.loc_oc_line, = self.local_ax.plot([], [], '.', color='orange')
        self.loc_big_line, = self.local_ax.plot([], [], '.', color='red')
        self.loc_car_pose, = self.local_ax.plot([], [], 'o', color=(1,0.4,0))
        self.loc_path_line, = self.local_ax.plot([], [], '-', color='red')
        self.loc_static_lines = [self.loc_car_pose]
        self.loc_dynamic_lines = [self.loc_yc_line, self.loc_bc_line, self.loc_oc_line, self.loc_big_line, self.loc_path_line]

    def set_state(self, state):
        self.cones_world = state["cones_world"]
        self.path = state["path"]
        self.start_pose = state["start_pose"]
        self.determine_map_bounds()
        self.draw_static_map()

    def determine_map_bounds(self):
        cones = self.cones_world.copy()
        cones[:,:2] = global_to_local(cones[:,:2], self.start_pose)
        offset = 5
        height = 6
        max_width = 19.0
        x_size = cones[:, 0].max() - cones[:, 0].min() + 2*offset
        y_size = cones[:, 1].max() - cones[:, 1].min() + 3*offset
        map_ratio = x_size/y_size
        width = height+(height*map_ratio)
        if max_width < width:
            map_ratio *= (max_width-height)/(width-height)
            width = max_width
        gs = GridSpec(1, 2, 
                        width_ratios=[map_ratio, 1], 
                        right=1-0.2/width, 
                        top=1, 
                        left=0.45/width, 
                        bottom=0.15/height, 
                        wspace=1.2/width)

        axes = [self.global_ax, self.local_ax]
        for i in range(2):
            ax = axes[i]
            ax.set_position(gs[i].get_position(self.fig))

        self.fig.set_size_inches(width,height)
        self.global_ax.set_xticks(np.arange(((cones[:, 0].min()-offset)//25)*25, ((cones[:, 0].max()+offset)//25+1)*25, 25)[1:])
        self.global_ax.set_xticks(np.arange(((cones[:, 0].min()-offset)//5)*5, ((cones[:, 0].max()+offset)//5+1)*5, 5)[1:], minor = True)
        self.global_ax.set_yticks(np.arange(((cones[:, 1].min()-offset)//25)*25, ((cones[:, 1].max()+offset*2)//25+1)*25, 25)[1:])
        self.global_ax.set_yticks(np.arange(((cones[:, 1].min()-offset)//5)*5, ((cones[:, 1].max()+offset*2)//5+1)*5, 5)[1:], minor = True)
        self.global_ax.set_xlim(cones[:, 0].min()-offset, cones[:, 0].max()+offset)
        self.global_ax.set_ylim(cones[:, 1].min()-offset, cones[:, 1].max()+offset*2)
        self.glob_text.set_position((cones[:, 0].min()-offset+0.5, cones[:, 1].max()+2.0*offset-0.5))
        self.global_ax.set_aspect('equal', adjustable='box')

    def draw_static_map(self):
        # reset data buffers
        for line in self.loc_dynamic_lines + self.map_dynamic_lines + self.loc_static_lines + self.map_static_lines: line.set_data([], [])

        self.glob_text.set_text("")
        self.map_car_pose.set_data([], [])
        plt.pause(0.0001)

        # draw static global map
        cones = self.cones_world.copy()
        # cones[:,:2] = global_to_local(cones[:,:2], self.start_pose)

        self.map_yc_line.set_data(*cones[cones[:, 2]==ConeClasses.YELLOW][:,0:2].T)
        self.map_bc_line.set_data(*cones[cones[:, 2]==ConeClasses.BLUE][:,0:2].T)
        self.map_oc_line.set_data(*cones[cones[:, 2]==ConeClasses.ORANGE][:,0:2].T)
        self.map_big_line.set_data(*cones[cones[:, 2]==ConeClasses.BIG][:,0:2].T)
        self.map_path_line.set_data(*self.path.T)
        for line in self.map_static_lines: self.global_ax.draw_artist(line)
        self.global_static_background = self.fig.canvas.copy_from_bbox(self.global_ax.bbox)

        # draw static local map
        self.loc_car_pose.set_data([0.], [0.])
        for line in self.loc_static_lines: self.local_ax.draw_artist(line)
        self.local_static_background = self.fig.canvas.copy_from_bbox(self.local_ax.bbox)

    def render_state(self, car_pose, text):
        # start = time.time()
        self.fig.canvas.restore_region(self.global_static_background)
        self.fig.canvas.restore_region(self.local_static_background)
        # obs = self.state.get_obs()

        # # plot text fast
        self.glob_text.set_text(text)
        self.global_ax.draw_artist(self.glob_text)

        # global state
        # if draw_extra is not None:
            # draw_extra()

        self.map_car_pose.set_data(car_pose[0], car_pose[1])
        for line in self.map_dynamic_lines[::-1]: self.global_ax.draw_artist(line)

        # local state
        cones = np.array([[2, 1, 0]])
        # cones = obs["percep_data"]
        self.loc_yc_line.set_data(*cones[cones[:, 2]==ConeClasses.YELLOW][:,0:2].T)
        self.loc_bc_line.set_data(*cones[cones[:, 2]==ConeClasses.BLUE][:,0:2].T)
        self.loc_oc_line.set_data(*cones[cones[:, 2]==ConeClasses.ORANGE][:,0:2].T)
        self.loc_big_line.set_data(*cones[cones[:, 2]==ConeClasses.BIG][:,0:2].T)
        # if path is not None: self.loc_path_line.set_data(path[:,0], path[:,1])
        for line in self.loc_dynamic_lines: self.local_ax.draw_artist(line)

        self.fig.canvas.blit(self.global_ax.bbox)
        self.fig.canvas.blit(self.local_ax.bbox)
        self.fig.canvas.flush_events()
        # print(f"fps: {1/(time.time()-start)}")
