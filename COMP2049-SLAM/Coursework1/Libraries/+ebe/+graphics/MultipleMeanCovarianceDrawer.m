classdef MultipleMeanCovarianceDrawer < ebe.graphics.Drawer

    % This drawer draws covariance mean and covariance matrices for
    % multiple 

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

        function obj = MultipleMeanCovarianceDrawer(colour, sigmaValue)

            if (nargin == 1)
                obj.sigmaValue = 2;
            else
                obj.sigmaValue = sigmaValue;
            end

            obj.angleStep = 5;

            obj.meanHandle = plot(NaN, NaN, '+', 'Color', colour);
            obj.ellipseHandle = plot(NaN, NaN, 'Color', colour, 'LineWidth', 1.5);

        end

        function update(obj, x, P)

            % Update the graphics handle for the mean values
            set(obj.meanHandle, 'XData', x(1, :), 'YData', x(2, :));

            % Compute the covariance ellipses; NaN marks the end of one
            % ellipse

            if (ndims(P) == 3)
                covPtsX = [];
                covPtsY = [];
                for p =  1 : size(P, 3)
                    ptsC = obj.covarianceEllipsePoints(x(:, p), P(:, :, p));
                    covPtsX = cat(2, covPtsX, [ptsC(1, :) NaN]);
                    covPtsY = cat(2, covPtsY, [ptsC(2, :) NaN]);  
                end
                % Update the graphics handle for all the covariance points
                set(obj.ellipseHandle, 'XData', covPtsX, 'YData', covPtsY);
            else
                pts = obj.covarianceEllipsePoints(x, P);
                set(obj.ellipseHandle, 'XData', pts(1, :), 'YData', pts(2, :));
            end
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