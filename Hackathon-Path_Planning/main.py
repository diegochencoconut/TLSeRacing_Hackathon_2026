from src.track import Track
from src.planner import PathPlanner
from src.visualizer import plot_track_and_path
import os
import numpy as np  

def main():
    TRACK_NAME = "peanut.csv"
    TRACK_PATH = os.path.join("tracks_data",TRACK_NAME)

    track = Track(TRACK_PATH)

    planner = PathPlanner()
    # path_simple = planner.simple_compute_path(track)
    # plot_track_and_path(track, path_simple)

    path_delaunay = planner.delaunay_triangulation_path(track)
    plot_track_and_path(track, path_delaunay)


if __name__ == "__main__":
    main()