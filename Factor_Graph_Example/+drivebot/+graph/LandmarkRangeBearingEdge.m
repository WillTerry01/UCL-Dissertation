% This edge encodes a 3D range bearing measurement.
%
% The measurement is in spherical polar coordinates

% Jacobian from https://github.com/petercorke/robotics-toolbox-matlab/blob/master/RangeBearingSensor.m

classdef LandmarkRangeBearingEdge < g2o.core.BaseBinaryEdge
    
    methods(Access = public)
    
        function this = LandmarkRangeBearingEdge()
            this = this@g2o.core.BaseBinaryEdge(2);
        end
        
        function initialize(this)
            % Q2b:
            % Complete implementation

            % Get current vehicle state
            x = this.edgeVertices{1}.x;

            % Initialise the measurement
            m = zeros(2, 1);

            % Compute landmark measurement based on vehicle state x[k] and
            % measurement associated with this edge
            m(1) = x(1) + this.z(1) * cos(this.z(2) + x(3));
            m(2) = x(2) + this.z(1) * sin(this.z(2) + x(3));

            % Set measurement matrix
            this.edgeVertices{2}.setEstimate(m);
        end
        
        function computeError(this)

            % Q2b:
            % Complete implementation


            % Get the current vehicle state measurement x, and the landmark
            % measurement m
            x = this.edgeVertices{1}.estimate();
            m = this.edgeVertices{2}.estimate();

            % Compute dx using first two rows of x and m
            dx = m(1:2) - x(1:2);

            % Compute error matrix in in terms of both range and bearing
            this.errorZ(1) = norm(dx) - this.z(1);
            this.errorZ(2) = g2o.stuff.normalize_theta(atan2(dx(2), dx(1)) - x(3) - this.z(2));

        end
        
        function linearizeOplus(this)
            % Q2b:
            % Complete implementation

            % Get the current vehicle state measurement x, and the landmark
            % measurement m
            x = this.edgeVertices{1}.estimate();
            m = this.edgeVertices{2}.estimate();

            % Compute dx using first two rows of x and m
            dx = m(1:2) - x(1:2);

            % Normalised distance for derivatives
            r = norm(dx);
            
            % J{1}=d(errorZ)/d(x[k])  vehicle state
            % J{2}=d(errorZ)/d(x[i])  landmark measurement
            this.J{1} = ...
                [-dx(1)/r -dx(2)/r 0;
                dx(2)/r^2 -dx(1)/r^2 -1];
            this.J{2} = - this.J{1}(1:2, 1:2);
        end        
    end
end