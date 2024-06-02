# load a .h5 file and visualize the data points

# fmt: off

import h5py
import numpy as np


# Load .h5 file containing 3d points. Return as numpy array

def load_h5_file(file_path):
    data = []
    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            data.append(f[key]['obs'][0][32:35])

    return np.array(data)

def visualize(data):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:, 0]*100, data[:, 1]*100, data[:, 2]*100)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def main():
    file_path = "trajectories.state.pd_joint_delta_pos.h5"
    data = load_h5_file(file_path)
    visualize(data)


if __name__ == "__main__":
    main()
