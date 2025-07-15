import h5py

h5_path = '/home/will/Dissertation/UCL-Dissertation/CPP-Working-Project/2D/H5_Files/2D_trajectory_estimate.h5'

with h5py.File(h5_path, 'r') as f:
    data = f['trajectory'][:]
    print(data.shape)
    print(data.dtype)

def load_h5_data(path, is_1d):
    with h5py.File(path, 'r') as f:
        data = f['trajectory'][:]
        if is_1d:
            # Columns: t, true_x, meas_x, est_x
            t = data[:, 0]
            x_true = data[:, 1]
            x_meas = data[:, 2]
            x_est = data[:, 3]
            return t, x_true, x_meas, x_est
        else:
            # Columns: t, true_x, true_y, meas_x, meas_y, est_x, est_y
            t = data[:, 0]
            x_true = data[:, 1]
            y_true = data[:, 2]
            x_meas = data[:, 3]
            y_meas = data[:, 4]
            x_est = data[:, 5]
            y_est = data[:, 6]
            return t, x_true, y_true, x_meas, y_meas, x_est, y_est