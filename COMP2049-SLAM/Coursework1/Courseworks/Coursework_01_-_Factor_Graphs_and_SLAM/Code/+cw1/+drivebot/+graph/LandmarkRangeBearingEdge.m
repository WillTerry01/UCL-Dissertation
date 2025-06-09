classdef LandmarkRangeBearingEdge < g2o.core.BaseBinaryEdge
    % LandmarkRangeBearingEdge summary of LandmarkRangeBearingEdge
    %
    % This class stores an edge which represents the factor for observing
    % the range and bearing of a landmark from the vehicle. Note that the
    % sensor is fixed to the platform.
    %
    % The measurement model is
    %
    %    z_(k+1)=h[x_(k+1)]+w_(k+1)
    %
    % The measurements are r_(k+1) and beta_(k+1) and are given as follows.
    % The sensor is at (lx, ly).
    %
    %    dx = lx - x_(k+1); dy = ly - y_(k+1)
    %
    %    r(k+1) = sqrt(dx^2+dy^2)
    %    beta(k+1) = atan2(dy, dx) - theta_(k+1)
    %
    % The error term
    %    e(x,z) = z(k+1) - h[x(k+1)]
    %
    % However, remember that angle wrapping is required, so you will need
    % to handle this appropriately in compute error.
    %
    % Note this requires estimates from two vertices - x_(k+1) and l_(k+1).
    % Therefore, this inherits from a binary edge. We use the convention
    % that vertex slot 1 contains x_(k+1) and slot 2 contains l_(k+1).
    
    methods(Access = public)
    
        function obj = LandmarkRangeBearingEdge()
            % LandmarkRangeBearingEdge for LandmarkRangeBearingEdge
            %
            % Syntax:
            %   obj = LandmarkRangeBearingEdge(landmark);
            %
            % Description:
            %   Creates an instance of the LandmarkRangeBearingEdge object.
            %   Note we feed in to the constructor the landmark position.
            %   This is to show there is another way to implement this
            %   functionality from the range bearing edge from activity 3.
            %
            % Inputs:
            %   landmark - (2x1 double vector)
            %       The (lx,ly) position of the landmark
            %
            % Outputs:
            %   obj - (handle)
            %       An instance of a ObjectGPSMeasurementEdge

            obj = obj@g2o.core.BaseBinaryEdge(2);
        end
        
        function initialEstimate(obj)
            % INITIALESTIMATE Compute the initial estimate of the landmark.
            %
            % Syntax:
            %   obj.initialEstimate();
            %
            % Description:
            %   Compute the initial estimate of the landmark given the
            %   platform pose and observation.

           % warning('LandmarkRangeBearingEdge.initialEstimate: implement')
            
            lx = obj.edgeVertices{1}.x(1) + ((obj.z(1) * cos(obj.edgeVertices{1}.x(3)) + obj.z(2)));
            ly = obj.edgeVertices{1}.x(2) + ((obj.z(1) * sin(obj.edgeVertices{1}.x(3)) + obj.z(2)));
            

            obj.edgeVertices{2}.x = [lx;ly];
        end
        
        function computeError(obj)
            % COMPUTEERROR Compute the error for the edge.
            %
            % Syntax:
            %   obj.computeError();
            %
            % Description:
            %   Compute the value of the error, which is the difference
            %   between the predicted and actual range-bearing measurement.
            
            
            dx = obj.edgeVertices{2}.x(1) - obj.edgeVertices{1}.x(1);
            dy = obj.edgeVertices{2}.x(2) - obj.edgeVertices{1}.x(2);
            phi = g2o.stuff.normalize_theta(obj.edgeVertices{1}.x(3));
            r_k1 = sqrt(dx^2 + dy^2);
            beta_k1 = g2o.stuff.normalize_theta(g2o.stuff.normalize_theta(atan2(dy, dx)) - phi);

            obj.errorZ(1) = r_k1 - obj.z(1);
            obj.errorZ(2) = g2o.stuff.normalize_theta(beta_k1 - obj.z(2));
        end

        
        function linearizeOplus(obj)
            % linearizeOplus Compute the Jacobian of the error in the edge.
            %
            % Syntax:
            %   obj.linearizeOplus();
            %
            % Description:
            %   Compute the Jacobian of the error function with respect to
            %   the vertex.
            %

            dx = obj.edgeVertices{2}.x(1) - obj.edgeVertices{1}.x(1);
            dy = obj.edgeVertices{2}.x(2) - obj.edgeVertices{1}.x(2);
            r = sqrt(dx^2 + dy^2);

            obj.J{1} = [-dx/r      -dy/r        0;
                        dy/(r^2)   -dx/(r^2)    -1];

            
            obj.J{2} = [    dx/r        dy/r;
                            -dy/(r^2)   dx/(r^2)];
        end        
    end
end