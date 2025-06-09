classdef SLAMSystem < ebe.core.EventBasedEstimator
    % SLAMSystem summary of SLAMSystem
    %
    % A SLAM system is one which builds a map of the environment and
    % estimates the platform state at the same time. The main thing this
    % class provides is an interface for extracting the estimates.
    
    methods(Access = public)
        function obj = SLAMSystem(configuration)
            % SLAMSystem Constructor for SLAMSystem
            %
            % Syntax:
            %   slamSystem = SLAMSystem(config)
            %
            % Description:
            %   Constructs a SLAM system using the provided configuration.
            %   Note this is an abstract class and cannot be instantiated
            %   directly. 
            %
            % Inputs:
            %   config - (struct)
            %       The configuration structure
            %
            % Outputs:
            %   slamSystem - (handle)
            %       An instance of a SLAMSystem
            obj@ebe.core.EventBasedEstimator(configuration);
        end
    end

    methods(Access = public, Abstract)

        [x, P] = platformEstimate(obj);
            % PLATFORMESTIMATE Return the current mean and covariance of the
            % platform estimate
            %
            % Syntax:
            %   [x, P] = slamSystem.platformEstimate()
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

        
        [T, X, PX] = platformEstimateHistory(this);

        [m, Pmm, landmarkIds] = landmarkEstimates(this);
            % LANDMARKESTIMATES Return the current mean and covariance of
            % each landmark estimate.
            %
            % Syntax:
            %   [m, Pmm, landmarkIds] = slamSystem.platformEstimate()
            %
            % Description:
            %   Return the current estimates of the landmarks and the IDs.
            %   The landmark covariances are just the blocks on the
            %   diagonals. The full landmark covariance block is not
            %   returned.
            %
            %   At a given time, there are Nk landmarks. The dimension of
            %   each landmark is l
            %
            % Outputs:
            %   m - (lxN_k vector)
            %       A column vector which contains the estimated mean of
            %       each landmark position.
            %   PX - (lxlxN_k double psd matrix)
            %       A three dimensional matrix which stores the landmark
            %       estimates. The covariance of landmark 4, for example,
            %       is given by P(:,:,4)
    end
end
