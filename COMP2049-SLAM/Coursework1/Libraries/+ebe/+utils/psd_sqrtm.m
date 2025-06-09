function S = psd_sqrtm(A)
    % Eigenvalue decomposition
    [V, D] = eig(A);
    
    % Ensure eigenvalues are non-negative (account for numerical precision)
    tol = 1e-12; % Tolerance for treating values as zero
    D(D < tol) = 0;
    
    % Compute the square root of the eigenvalues
    sqrtD = sqrt(D);
    
    % Reconstruct the square root matrix
    S = V * sqrtD;
end
