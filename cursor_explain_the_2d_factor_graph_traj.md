# Explain the 2D factor graph trajectory file
_Exported on 24/06/2025 at 15:39:13 BST from Cursor (1.0.0)_

---

**User**

Could you go through and explain the 2D_factor_graph_trajectory file to me in great detail

---

**Cursor**

Of course! I'd be happy to walk you through the `2D_factor_graph_trajectory.cpp` file. It's an excellent example of how to solve a state estimation problem using a factor graph optimization library called `g2o`.

Let's break it down section by section.

### 1. Overview & Goal

The primary goal of this C++ program is to estimate the 2D trajectory (position and velocity) of a moving object over time. It does this by:
1.  **Simulating a "true" trajectory**: It creates a ground truth path for an object (e.g., moving in a straight line, circle, or square).
2.  **Simulating noisy measurements**: It generates sensor-like position measurements at each time step, which are corrupted with random noise.
3.  **Building a Factor Graph**: It constructs a mathematical model of the problem using vertices (representing the unknown states) and edges (representing constraints from the motion model and measurements).
4.  **Optimizing the Graph**: It uses the `g2o` library to find the most likely sequence of states (the estimated trajectory) that best fits the measurements and the motion model.
5.  **Analyzing Consistency**: It calculates statistical metrics called **CNEES** and **CNIS** to check if the uncertainty (covariance) produced by the optimizer is consistent with the actual errors.
6.  **Saving Results**: It saves the true trajectory, estimated trajectory, and consistency metrics to a `.csv` file, which is then used by `plot_trajectory.py` to visualize the results.

---

### 2. Core Components: Vertices and Edges

Factor graphs are built from two main components: vertices and edges.

#### `Vertex4D` (The State)
```cpp
// ...
class Vertex4D : public BaseVertex<4, Eigen::Vector4d> {
// ...
};
```
*   **Purpose**: This class defines a **vertex** in our graph. Each vertex represents the state of the object at a single point in time.
*   **State Vector**: The state is a 4-dimensional vector `[x, y, vx, vy]`, representing the 2D position (`x`, `y`) and 2D velocity (`vx`, `vy`).
*   **`_estimate`**: This is the most important member variable. It holds the current best guess for the state vector of that vertex. The optimizer's job is to adjust the `_estimate` of all vertices to minimize the overall error in the graph.

#### `EdgeProcessModel` (The Motion Model)
```cpp
// ...
class EdgeProcessModel : public BaseBinaryEdge<4, Eigen::Vector4d, Vertex4D, Vertex4D> {
// ...
};
```
*   **Purpose**: This class defines an **edge** that connects two consecutive state vertices (e.g., the state at time `k` and the state at time `k+1`). It enforces the "rules of motion".
*   **`computeError()`**: This function is the core of the edge. It calculates the error based on a **constant velocity model**. It predicts the position at the next time step (`v2`) based on the current one (`v1`) and assumes the velocity remains the same.
    ```cpp
    // Predict v2 from v1 using constant velocity model
    Eigen::Vector4d pred;
    pred.head<2>() = v1->estimate().head<2>() + v1->estimate().tail<2>() * dt; // pos_k+1 = pos_k + vel_k * dt
    pred.tail<2>() = v1->estimate().tail<2>(); // vel_k+1 = vel_k
    _error = v2->estimate() - pred; // The error is the difference between the actual v2 and the prediction
    ```
*   **Information Matrix**: In the `main` function, this edge is given an information matrix `Q.inverse()`. The information matrix is the inverse of the covariance matrix. It tells the optimizer how much to "trust" this edge. A high information value means we are very certain about our motion model, and the optimizer will penalize deviations from it more heavily. `Q` represents the process noise, which accounts for unmodeled dynamics (like small accelerations).

#### `EdgeMeasurement` (The Measurement Model)
```cpp
// ...
class EdgeMeasurement : public BaseUnaryEdge<2, Eigen::Vector2d, Vertex4D> {
// ...
};
```
*   **Purpose**: This class defines an edge that connects a single state vertex to an external piece of information—in this case, a noisy position measurement.
*   **`computeError()`**: This function calculates the difference between the position part of the vertex's state (`v->estimate().head<2>()`) and the actual sensor measurement (`_measurement`).
*   **Information Matrix**: This edge is given the information matrix `R.inverse()`. `R` represents the measurement noise covariance. It tells the optimizer how much to trust the sensor measurements. If `R` is small (so `R.inverse()` is large), the optimizer will try very hard to make the estimated trajectory pass close to the measurement points.

---

### 3. The `main` Function: Putting It All Together

This is where the simulation and optimization happen.

1.  **Setup and Trajectory Generation (Lines 70-138)**:
    *   It initializes parameters like the number of steps `N`, initial state, and noise matrices `Q` and `R`.
    *   It prompts the user to choose a trajectory type (line, circle, or square).
    *   It then enters a loop to generate the `true_states` vector. Depending on the `traj_type`, it updates the position and velocity according to the desired motion. Crucially, it adds a small amount of `process_noise` at each step to make the simulation more realistic.
    *   It also generates a `measurements` vector by taking the true position at each step and adding `measurement_noise`.

2.  **Optimizer Setup (Lines 141-147)**:
    *   This section contains standard boilerplate code for setting up the `g2o` optimizer. It configures a Levenberg-Marquardt solver, which is a common and robust algorithm for non-linear least squares problems like this one.

3.  **Graph Construction (Lines 150-176)**:
    *   **Vertices**: It creates `N` instances of `Vertex4D`, one for each time step. Each vertex is given an initial guess for its state, which is just the true state plus some extra noise.
    *   **Process Edges**: It creates `N-1` `EdgeProcessModel` edges, linking `vertex[k-1]` to `vertex[k]` for all `k` from 1 to `N`. This builds the temporal chain of the trajectory.
    *   **Measurement Edges**: It creates `N` `EdgeMeasurement` edges. Each edge connects a vertex `vertex[k]` to its corresponding measurement `measurements[k]`.

4.  **Optimization (Lines 179-180)**:
    *   `optimizer.initializeOptimization()` prepares the graph.
    *   `optimizer.optimize(20)` runs the solver for 20 iterations. The solver iteratively adjusts the `_estimate` in all vertices to minimize the sum of squared errors from all the edges in the graph.

5.  **Consistency Analysis: CNEES and CNIS (Lines 183-214)**:
    *   This is a crucial step for validating the filter/optimizer.
    *   **CNEES (Consistent Normalized Estimation Error Squared)**: This metric checks if the actual estimation error `(estimate - true_state)` is consistent with the optimizer's own calculated uncertainty (the covariance matrix `P`).
        *   `optimizer.computeMarginals(...)` calculates the covariance matrix `P` for a given state.
        *   The NEES is calculated as `error^T * P^-1 * error`.
        *   **What it means**: If the optimizer is well-tuned, the *average* CNEES value should be approximately equal to the dimension of the state (which is 4 here). If it's much larger, the optimizer is too optimistic (its calculated uncertainty is too small). If it's smaller, the optimizer is too pessimistic.
    *   **CNIS (Consistent Normalized Innovation Squared)**: This checks if the innovation (the difference between the actual measurement `z` and the predicted measurement `H*estimate`) is consistent with the innovation covariance `S = H*P*H^T + R`.
        *   **What it means**: Similar to CNEES, the *average* CNIS value should be approximately equal to the dimension of the measurement (which is 2 here). It's another way to check if the filter's uncertainty modeling is correct.

6.  **Output Results (Lines 217-234)**:
    *   The code opens a CSV file.
    *   It then loops through all the time steps and writes the true state, the final optimized estimated state, the CNEES value, and the CNIS value for that step into a new row in the file. This file is what `plot_trajectory.py` reads to create all the plots.

I hope this detailed breakdown helps you understand the code! Let me know if any part is still unclear.

```cpp
// ...
class Vertex4D : public BaseVertex<4, Eigen::Vector4d> {
// ...
};
```

```cpp
// ...
class EdgeProcessModel : public BaseBinaryEdge<4, Eigen::Vector4d, Vertex4D, Vertex4D> {
// ...
};
```

```cpp
    // Predict v2 from v1 using constant velocity model
    Eigen::Vector4d pred;
    pred.head<2>() = v1->estimate().head<2>() + v1->estimate().tail<2>() * dt; // pos_k+1 = pos_k + vel_k * dt
    pred.tail<2>() = v1->estimate().tail<2>(); // vel_k+1 = vel_k
    _error = v2->estimate() - pred; // The error is the difference between the actual v2 and the prediction
```

```cpp
// ...
class EdgeMeasurement : public BaseUnaryEdge<2, Eigen::Vector2d, Vertex4D> {
// ...
};
```

