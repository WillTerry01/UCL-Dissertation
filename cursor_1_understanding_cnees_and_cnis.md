# (1) Understanding CNEES and CNIS values
_Exported on 25/06/2025 at 19:22:12 BST from Cursor (1.0.0)_

---

**User**

I have currently set up my project so that it calculates the chi2 values. The next goal is related to the CNEES and CNIS Values. Can you explain to me where these would be implemented/ what they mean

---

**Cursor**

Certainly! Here’s an explanation of **CNEES** and **CNIS** values, what they mean, and where you would typically implement their calculation in your project.

---

## What are CNEES and CNIS?

### 1. **CNEES**: Consistency of Normalized Estimation Error Squared
- **Definition**: CNEES is a statistical metric used to evaluate the consistency of an estimator (e.g., a Kalman filter or a trajectory estimator).
- **How it works**:  
  - For each time step, you compute the estimation error (difference between the estimated state and the true state).
  - You normalize this error by the estimated covariance (uncertainty) of your estimator.
  - The squared normalized error is then compared to the expected value (which should be close to the dimension of the state if the estimator is consistent).
- **Formula** (for a state vector \( x \) of dimension \( n \)):
  \[
  \text{NEES}_k = (x_k - \hat{x}_k)^T P_k^{-1} (x_k - \hat{x}_k)
  \]
  where:
  - \( x_k \) = true state at time \( k \)
  - \( \hat{x}_k \) = estimated state at time \( k \)
  - \( P_k \) = estimated covariance at time \( k \)
- **CNEES** is the average NEES over all time steps and/or Monte Carlo runs.

### 2. **CNIS**: Consistency of Normalized Innovation Squared
- **Definition**: CNIS is similar to CNEES, but it evaluates the consistency of the innovation (the difference between the predicted and actual measurement).
- **How it works**:
  - For each measurement, you compute the innovation (residual).
  - You normalize this by the innovation covariance.
  - The squared normalized innovation is compared to the expected value (should be close to the measurement dimension if consistent).
- **Formula** (for measurement \( z \) of dimension \( m \)):
  \[
  \text{NIS}_k = (z_k - \hat{z}_k)^T S_k^{-1} (z_k - \hat{z}_k)
  \]
  where:
  - \( z_k \) = actual measurement at time \( k \)
  - \( \hat{z}_k \) = predicted measurement at time \( k \)
  - \( S_k \) = innovation covariance at time \( k \)
- **CNIS** is the average NIS over all time steps and/or Monte Carlo runs.

---

## Where to Implement in Your Project

### **Typical Implementation Points**
- **CNEES**:  
  - After each state update (i.e., after you have an updated state estimate and its covariance), compute the NEES for that time step.
  - Accumulate these values to compute the average (CNEES) at the end of the run or over multiple runs.
- **CNIS**:  
  - After each measurement update (i.e., after you compute the innovation and its covariance), compute the NIS for that time step.
  - Accumulate these values to compute the average (CNIS) at the end of the run or over multiple runs.

### **In Your Codebase**
Given your files, the most likely places to implement these are:
- **1D_factor_graph_trajectory.cpp / .h**:  
  - If this is where the state estimation and measurement updates happen, you would add NEES and NIS calculations here.
- **1D_single_run.cpp** or **1D_main.cpp**:  
  - If you run multiple simulations or aggregate results, you would accumulate and report the CNEES and CNIS here.

---

## **Summary Table**

| Metric | What | Where to Compute | Where to Aggregate/Report |
|--------|------|------------------|--------------------------|
| CNEES  | Consistency of Normalized Estimation Error Squared | After each state update | At the end of run(s) |
| CNIS   | Consistency of Normalized Innovation Squared | After each measurement update | At the end of run(s) |

---

## **Next Steps**
- **Identify** where your code updates the state and processes measurements.
- **Insert** NEES and NIS calculations at those points.
- **Aggregate** the results and output the averages at the end.

Would you like help identifying the exact lines or functions in your code where these calculations should go? If so, I can look into your `1D_factor_graph_trajectory.cpp` and related files to point out the precise spots!

