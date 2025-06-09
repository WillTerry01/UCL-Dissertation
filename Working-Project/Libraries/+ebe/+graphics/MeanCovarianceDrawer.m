classdef MeanCovarianceDrawer < ebe.graphics.Drawer

    properties(Access = protected)
        % The handle of the graphics representing the mean
        meanHandle;

        % The handle of the graphics representing the orientation
        ellipseHandle;

        % The sigma ellipse value used, corresponding to the standard
        % deviation number
        sigmaValue;

        % The angular difference between each point on the 
        angleStep;
    end


    methods(Access = public)

        function obj = MeanCovarianceDrawer(colour, sigmaValue)

            if (nargin == 1)
                obj.sigmaValue = 2;
            else
                obj.sigmaValue = sigmaValue;
            end

            obj.angleStep = 5;

            obj.meanHandle = plot(NaN, NaN, 'Color', colour, 'Marker', '+');
            obj.ellipseHandle = plot(NaN, NaN, 'Color', colour, 'LineWidth', 1.5);

        end

        function update(obj, x, P)
            set(obj.meanHandle, 'XData', x(1), 'YData', x(2));
            pts = obj.covarianceEllipsePoints(x, P);
            set(obj.ellipseHandle, 'XData', pts(1, :), 'YData', pts(2, :));
        end
    end

    methods(Access = protected)
        function ptsC = covarianceEllipsePoints(obj, x, P)
        
            % Handle non-PSD
            if (min(eig(P)) < 0)
                ptsC = zeros(2, 0);
                return
            end

            % Sample points
            theta = deg2rad(0 : obj.angleStep : 360);
            ptsC = obj.sigmaValue * sqrtm(P+1e-12*eye(2)) * [cos(theta);sin(theta)];
            
            %translate
            ptsC = ptsC + x(:);
        end 
    end
end