classdef EstimatorView < ebe.graphics.View
    % EstimatorView summary of EstimatorView
    %
    % Subclasses of this object are used to visualize an estimator

    properties(Access = protected)

        % The estimator to be visualized
        estimator;        
    end

    methods(Access = public)

        function obj = EstimatorView(config, estimator)
            % EstimatorView Constructor for EstimatorView
            %
            % Syntax:
            %   estimatorView = EstimatorView(config, estimator);
            %
            % Description:
            %   Constructor for an instance of a View object for an
            %   estimator.
            %
            % Inputs:
            %   config - (struct)
            %       The configuration structure
            %   estimator - (ebe.core.EventBasedEstimator)
            %       The estimator generator to be visualized

            obj@ebe.graphics.View(config);

            obj.estimator = estimator;
        end
            
    end
end