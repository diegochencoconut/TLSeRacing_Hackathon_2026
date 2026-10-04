from src.track import Track
import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt

class PathPlanner:
    def __init__(self, step_size : float = 0.5, smoothing : float = 1.0):
        self.step_size = step_size
        self.smoothing = smoothing

    def simple_compute_path(self, track : Track):

        start_coo = (track.car_start.x, track.car_start.y)
        path = [start_coo]

        # Simplest track: just the center between a pair of cones (blue and yellow)
        for blue_cone, yellow_cone in zip(track.blue_cones, track.yellow_cones):
            center_x = (blue_cone.x + yellow_cone.x) / 2
            center_y = (blue_cone.y + yellow_cone.y) / 2
            path.append((center_x, center_y))

        return path

    def delaunay_triangulation_path(self, track : Track):

        start_coo = (track.car_start.x, track.car_start.y)
        path = [start_coo]

        blue_cones = np.array([(cone.x, cone.y) for cone in track.blue_cones])
        yellow_cones = np.array([(cone.x, cone.y) for cone in track.yellow_cones])

        plt.figure()
        plt.scatter(blue_cones[:, 0], blue_cones[:, 1], color='blue', label='Blue Cones')
        plt.scatter(yellow_cones[:, 0], yellow_cones[:, 1], color='yellow', label='Yellow Cones')
        plt.legend()
        plt.title("Cones for Delaunay Triangulation")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # Triangulate each successive group of five cones from each boundary.
        BATCH_SIZE = 10
        batch_count = min(len(blue_cones), len(yellow_cones))
        start = 0
        while start < batch_count:
            blue_batch = blue_cones[start:start + BATCH_SIZE]
            yellow_batch = yellow_cones[start:start + BATCH_SIZE]
            # print(f"Processing batch: start={start}, blue_batch={blue_batch}, yellow_batch={yellow_batch}")

            if len(blue_batch) < BATCH_SIZE:
                blue_batch = np.vstack((blue_batch, blue_cones[0:BATCH_SIZE - len(blue_batch)]))

            if len(yellow_batch) < BATCH_SIZE:
                yellow_batch = np.vstack((yellow_batch, yellow_cones[0:BATCH_SIZE - len(yellow_batch)]))


            start = start + BATCH_SIZE - 1      # to make sure we have some overlap between batches
            points = np.vstack((blue_batch, yellow_batch))

            if len(points) < 3:
                continue

            try:
                delaunay = Delaunay(points)
            except Exception:
                continue

            plt.triplot(points[:, 0], points[:, 1], delaunay.simplices,
                        color='gray', alpha=0.5)

            blue_count = len(blue_batch)
            edges = []
            for simplex in delaunay.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        p1, p2 = simplex[i], simplex[j]
                        if (p1 < blue_count) != (p2 < blue_count):
                            edges.append(tuple(sorted((p1, p2))))

            for p1, p2 in sorted(set(edges)):
                midpoint = (points[p1] + points[p2]) / 2
                path.append(tuple(midpoint))
            

        # Plot the path
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], color='red', label='Planned Path', linewidth=2)
        plt.scatter(path[:, 0], path[:, 1], color='red', label='Searched Points', s=10)
        plt.legend()

        # plt.show()

        return path


