classdef BaseUnaryEdge < g2o.core.BaseEdge
    % BaseUnaryEdge summary of BaseUnaryEdge
    % An edge which connects to a single vertex.

    methods(Access = protected)
   
        function obj = BaseUnaryEdge(measurementDimension)
            % BaseUnaryEdge Constructor for BaseUnaryEdge
            %
            % Syntax:
            %   obj = BaseUnaryEdge(dimension)
            %
            % Description:
            %   Creates a vertex object which only connects to a single
            %   vertex.
            %
            % Inputs:
            %   measurementDimension - (int)
            %       The measurement dimension stored in the state.
            %
            % Outputs:
            %   obj - (handle)
            %       An instance of a BaseUnaryEdge
            %
            % See Also:
            %   BASEEDGE

            obj = obj@g2o.core.BaseEdge(1, measurementDimension);
        end
    end
end
