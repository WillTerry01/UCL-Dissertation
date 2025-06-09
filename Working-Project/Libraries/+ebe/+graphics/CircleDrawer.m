classdef CircleDrawer < ebe.graphics.Drawer

    properties(Access = protected)

        radius;
        rectanglePosition;
        circleGeometry;

    end

    methods(Access = public)

        function obj = CircleDrawer(colour, radius)

            if (nargin == 1)
                radius = 0.2;
            end

            obj.radius = radius;
            d = 2 * radius;

            % Draw initial graphic; note rectangle doesn't take NaNs to
            % mean "don't draw"; therefore we start the position way
            % outside of the axes so it doesn't initially appear
            obj.rectanglePosition = [-1e4 -1e4 d d];
            obj.circleGeometry = rectangle('Position',obj.rectanglePosition, 'Curvature',[1,1], 'FaceColor', colour);
        end

        function update(obj, x, ~)
            p = x - obj.radius;
            obj.rectanglePosition(1:2) = p';
            set(obj.circleGeometry, 'Position', obj.rectanglePosition);
        end
    end
end