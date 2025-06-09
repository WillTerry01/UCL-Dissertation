classdef OrientedTriangleDrawer < ebe.graphics.Drawer

    properties(Access = protected)

        trianglePoints;
        triangleGeometry;

    end

    methods(Access = public)

        function obj = OrientedTriangleDrawer(colour, scale)

            if (nargin == 1)
                scale = 1.1;
            end

            % Create the triangle associated with the ground truth robot
            obj.trianglePoints = scale * [0 -2.5 -2.5; 0 -1.2 1.2];
            obj.triangleGeometry = patch(obj.trianglePoints(1,:), ...
                obj.trianglePoints(2,:), colour, 'FaceAlpha', 0.5);
        end
        
        function update(obj, x)
            R = [cos(x(3)) -sin(x(3)); sin(x(3)) cos(x(3))];
            ptsG = R * obj.trianglePoints + x(1:2);
            set(obj.triangleGeometry, 'XData', ptsG(1,:), 'YData', ptsG(2,:));
        end
    end
end