classdef MeanCovarianceView < ebe.graphics.EstimatorView

    properties(Access = protected)

        % Estimated mean and covariance
        xpDrawer;

        % States used to compute mean and covariance
	    statesIdx;

    end

    methods(Access = public)

      function obj = MeanCovarianceView(config, estimator, statesIdx)
            obj@ebe.graphics.EstimatorView(config, estimator);
            obj.statesIdx = statesIdx;

        end

        function start(obj)
            if (isempty(obj.estimator.name()) == false)
                searchName = [obj.estimator.name(), '_xp_graphic'];
            else
                searchName = '';
            end
            colour = ebe.graphics.DistinguishableColours.assignColour(searchName);
            obj.xpDrawer = ebe.graphics.MeanCovarianceDrawer(colour);
        end

        function visualize(obj, ~)
            % Plot the raw particles
            [x, P] = obj.estimator.computeXP();
            obj.xpDrawer.update(x(obj.statesIdx),P(obj.statesIdx, obj.statesIdx));
        end
    end
end
