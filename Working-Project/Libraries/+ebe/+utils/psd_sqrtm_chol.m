function L = psd_sqrtm_chol(A)
    % Check if the matrix is symmetric
    if ~isequal(A, A')
        error('Input matrix must be symmetric.');
    end
    
    % Add a small positive diagonal if necessary (regularization)
    tol = 1e-12; % Tolerance for numerical issues
    A = (A + A') / 2; % Ensure symmetry
    [R, p] = chol(A + tol * eye(size(A))); % Try Cholesky decomposition
    
    if p > 0
        error('Matrix is not positive semidefinite within numerical tolerance.');
    end

    % Return lower triangular part (or R' if you need the upper part)
    L = R';
end
