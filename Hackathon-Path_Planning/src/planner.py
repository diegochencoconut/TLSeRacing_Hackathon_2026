from src.track import Track
import numpy as np
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt
from scipy.interpolate import make_splprep
from scipy.optimize import minimize

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
        # path = [start_coo]
        path = []

        blue_cones = np.array([(cone.x, cone.y) for cone in track.blue_cones])
        yellow_cones = np.array([(cone.x, cone.y) for cone in track.yellow_cones])

        plt.figure(figsize=(10, 6))
        plt.scatter(blue_cones[:, 0], blue_cones[:, 1], color='blue')
        plt.scatter(yellow_cones[:, 0], yellow_cones[:, 1], color='gold')

        # Triangulate each successive group of five cones from each boundary.
        BATCH_SIZE = 5
        batch_count = min(len(blue_cones), len(yellow_cones))
        start = 0
        while start < batch_count:
            blue_batch = blue_cones[start:start + BATCH_SIZE]
            yellow_batch = yellow_cones[start:start + BATCH_SIZE]
            # print(f"Processing batch: start={start}, blue_batch={blue_batch}, yellow_batch={yellow_batch}")

            # If the batch is smaller than BATCH_SIZE, 
            # we can wrap around to the beginning of the cones list to ensure we have enough points for triangulation.
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
                # print(p1, p2)
                midpoint = (points[p1] + points[p2]) / 2
                # print(f"Midpoint: {midpoint}")

                # Remove first path since it will be the same as the last path of last window
                if len(path) != 0 and np.allclose(midpoint, path[-1]):
                    continue
                # Check if we already made a circle
                if len(path) != 0 and np.allclose(midpoint, path[0]):
                    break
                path.append(tuple(midpoint))
            

        # We now try to smooth the path by spline
        path.append(path[0])  # Close the loop
        original_path = np.array(path)
        # print(path)
        spl, u = make_splprep(np.array(path).T, s=5, bc_type='periodic')

        # Obtain the path points from the spline representation
        path = spl(u)
        path = np.array(path).T
        smoothed_original_path = path

        plt.plot(original_path[:, 0], original_path[:, 1], color='green', label='Original Path', linewidth=1, linestyle='--')
        plt.scatter(original_path[:, 0], original_path[:, 1], color='green', label='Reference Points', s=10)
        plt.legend()
        plt.title("Results of Delaunay Triangulation")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.savefig('./report/delaunay_original.png', dpi=800)

        # plot the smoothed path

        plt.figure(figsize=(10, 6))
        plt.scatter(blue_cones[:, 0], blue_cones[:, 1], color='blue')
        plt.scatter(yellow_cones[:, 0], yellow_cones[:, 1], color='gold')
        plt.title("Smoothed Central Line Trajectory")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")        
        plt.plot(original_path[:, 0], original_path[:, 1], color='green', label='Original Path', linewidth=1, linestyle='--')
        plt.plot(path[:, 0], path[:, 1], color='green', linestyle='-', label='Smoothed Path', linewidth=1)
        plt.legend()
        plt.savefig('./report/delaunay_smoothed.png', dpi=800)

        # We now calculate the derivative of the spline to get the tangent vector at each point
        spl_derivative = spl.derivative()
        tangent_vectors = spl_derivative(u) / np.linalg.norm(spl_derivative(u), axis=0)
        normal_vectors = np.array([-tangent_vectors[1], tangent_vectors[0]])

        # We now aim to minimize the summation of curvature along the path.

        # Optimize one lateral offset per sampled path point.  This keeps the
        # optimized path in the local track corridor while minimizing its
        # discrete curvature.
        corridor = np.empty((len(path), 2))
        for i, point in enumerate(path):
            blue_index = np.argmin(np.linalg.norm(blue_cones - point, axis=1))
            yellow_index = np.argmin(np.linalg.norm(yellow_cones - point, axis=1))
            corridor[i, 0] = abs(np.dot(
                blue_cones[blue_index] - point, normal_vectors[:, i]))
            corridor[i, 1] = abs(np.dot(
                yellow_cones[yellow_index] - point, normal_vectors[:, i]))

        margin = 1  # meters
        lower = -(corridor[:, 1] - margin)
        upper = corridor[:, 0] - margin
        # Degenerate or too-narrow corridors keep the corresponding point at
        # its original location rather than producing invalid bounds.
        invalid = lower > upper
        lower[invalid] = 0.0
        upper[invalid] = 0.0

        def curvature_cost(offsets):
            candidate = path + offsets[:, None] * normal_vectors.T
            second_difference = (
                np.roll(candidate, -1, axis=0)
                - 2.0 * candidate
                + np.roll(candidate, 1, axis=0)
            )
            return (np.sum(second_difference ** 2)
                    + 1e-3 * np.sum(offsets ** 2))

        result = minimize(
            curvature_cost,
            np.zeros(len(path)),
            method='SLSQP',
            bounds=list(zip(lower, upper)),
            options={'maxiter': 300, 'ftol': 1e-8},
        )
        if result.success:
            path = path + result.x[:, None] * normal_vectors.T

        # modify the last point to be the same as the first point to close the loop
        path[-1] = path[0]

        plt.figure(figsize=(10, 6))
        plt.scatter(blue_cones[:, 0], blue_cones[:, 1], color='blue')
        plt.scatter(yellow_cones[:, 0], yellow_cones[:, 1], color='gold')
        plt.scatter(path[:, 0], path[:, 1], color='red', label='Optimized Points', s=10)
        plt.scatter(smoothed_original_path[:, 0], smoothed_original_path[:, 1], color='green', label='Smoothed Original Points', s=10)
        plt.title("Curvature Minimized Path")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")        
        plt.plot(smoothed_original_path[:, 0], smoothed_original_path[:, 1], color='green', linestyle='-', label='Original Path', linewidth=1)
        plt.plot(path[:, 0], path[:, 1], color='red', linestyle='-', label='Optimized Path', linewidth=1)
        plt.legend()
        plt.savefig('./report/optimized_path.png', dpi=800)

        optimized_path = path

        # do spline again for smoothing
        spl, u = make_splprep(path.T, s=5, bc_type='periodic')
        path = spl(u)
        path = np.array(path).T

        # print(f"Tangent vectors: {tangent_vectors}")
        # print(f"Normal vectors: {normal_vectors}")


        # Add back the starting point
        # path = np.vstack((start_coo, path))
        # print(f"Final path: {path}")

        plt.figure(figsize=(10, 6))
        plt.scatter(blue_cones[:, 0], blue_cones[:, 1], color='blue')
        plt.scatter(yellow_cones[:, 0], yellow_cones[:, 1], color='gold')
        plt.title("Final Smoothed Optimized Path")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")        
        plt.plot(optimized_path[:, 0], optimized_path[:, 1], color='red', linestyle='dotted', label='Original Path', linewidth=1)
        plt.legend()
        plt.plot(path[:, 0], path[:, 1], color='red', label='Final Path', linewidth=1.5)
        plt.legend()
        plt.savefig('./report/final_path.png', dpi=800)

        # plt.show()

        # add back the starting point
        path = np.vstack((start_coo, path))

        return path


