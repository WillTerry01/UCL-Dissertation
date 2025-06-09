classdef LocalizationSystem < ebe.core.EventBasedEstimator
    % LocalizationSystem summary of LocalizationSystem
    %
    % A localization system is an estimator which returns the estimate of a
    % platform. It does not separately handle estimates associated with
    % landmarks. The main thing this class provides is an interface for
    % extracting the estimates.

    methods(Access = public)
        function obj = LocalizationSystem(config)
            % LocalizationSystem Constructor for LocalizationSystem
            %
            % Syntax:
            %   localizationSystem = LocalizationSystem(config)
            %
            % Description:
            %   Constructs a localization system using the configuration. Note
            %   this is an abstract class and cannot be instantiated
            %   directly. The main thing this class offers is a public
            %   interface to access the platform estimate.
            %
            % Inputs:
            %   config - (struct)
            %       The configuration structure
            %
            % Outputs:
            %   localizationSystem - (handle)
            %       An instance of a LocalizationSystem

            obj@ebe.core.EventBasedEstimator(config);
        end
    end

    methods(Access = public, Abstract)

        [T, X, PX] = estimateHistory(obj);
            % ESTIMATEHISTORY Return the history of the estimated state
            %
            % Syntax:
            %   [T, X, PX] = localizationSystem.estimateHistory()
            %
            % Description:
            %   Return a history of the estimate time, the state, and the
            %   diagonals of the covariance matrix. This interface is
            %   deprecated in favour of computeXP and result accumulators
            %
            % Outputs:
            %   T - (Nx1 double array)
            %       The timestamps of all the estimates for N timesteps
            %   X - (nxN double array)
            %       For each timestep, the state vector
            %   PX - (nxN double array)
            %       For each timestep, the diagonal of the covariance
            %       matrix
            %
            % See also:
            %       XPACCUMULATOR
            
        [x, P] = computeXP(obj);
            % COMPUTEXP Return the current mean and covariance of the
            % estimate
            %
            % Syntax:
            %   [x, P] = localizationSystem.computeXP()
            %
            % Description:
            %   Return the current mean and covariance estimate of the
            %   platform. Note that the covariance estimate is the full
            %   covariance matrix
            %
            % Outputs:
            %   X - (nx1 vector)
            %       The estimated state at the current time
            %   PX - (nxn double psd matrix)
            %       The covariance matrix of the estimate

    end

end