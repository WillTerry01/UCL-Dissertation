function [Fd, Qd, Bd] = continuousToDiscrete(Fc, Qc, Bc, dT)
    % CONTINUOUSTODISCRETE Compute a constant, linear, continuous time
    % state space model to a discrete one.
    %
    % Syntax:
    %   [Fd, Qd] = continuousToDiscrete(Fc, Qc, dT);
    %   [Fd, Qd, Bd] = continuousToDiscrete(Fc, Qc, Bc, dT);
    %
    % Description:
    %   Converts a continuous time linear model of the form
    %
    %     x_dot(t) = Fc * x(t) + Bc * u(t) + v(t)
    %
    %   where v(t) is a zero-mean random variable with covariance Qc,
    %   to the continuous time model
    %
    %     x(k+1) = Fd * x(k) + Bd * u(k+1) + v(k+1)
    %
    %   where dT = t(k+1)-t(k) and v(k+1) s a zero-mean random variable
    %   with covariance Qd using van Loan's method.
    %
    %   Note that B is optional and can be ignored if there are no control
    %   inputs.
    %
    % Inputs:
    %   Fc - (double nxn matrix)
    %       The continuous time state transition matrix
    %   Qc - (double nxn matrix)
    %       The continuous time process noise
    %   Bc - (double nxu optional)
    %       The continuous time control input matrix.
    %   dT - (double)
    %       The time step length
    %
    % Outputs:
    %   Fd - (double nxn matrix)
    %       The discrete time state transition matrix
    %   Qc - (double nxn matrix)
    %       The discrete time process noise
    %   Bc - (double nxu optional)
    %       The discrete time control input matrix.

    % Use Van Loan's algorithm to work out a discrete => continuous
    % transformation. If Bc is passed in and Bd is requested, the control input
    % matrix will be converted as well
    
    % This kludgy logic is used because Bd is optional
    if (nargin == 3)
        dT = Bc;
    end
    
    n = size(Fc, 1);
    
    bigA = zeros(2*n);
    
    bigA(1:n,1:n) = -Fc * dT;
    bigA(1:n, n+1:end) = Qc * dT;
    bigA(n+1:end, n+1:end) = Fc' * dT;
    
    bigB=expm(bigA);
    
    Fd = bigB(n+1:end, n+1:end)';
    Qd = Fd * bigB(1:n, n+1:end);
    
    % Compute Bd if requested
    Bd = [];
    
    if (nargin == 4)
        nu = size(Bc, 2);
        bigA = zeros(n + nu);
        bigA(1:n,1:n) = Fc * dT;
        bigA(1:n, n+1:end) = Bc * dT;
        bigB=expm(bigA);
        Bd = bigB(1:n, n+1:end);
    end
end