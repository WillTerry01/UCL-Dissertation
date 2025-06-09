% This class specializes a BaseEdge to the case that it plugs into two
% vertices.

classdef BaseBinaryEdge < g2o.core.BaseEdge

    methods(Access = protected)
   
        function obj = BaseBinaryEdge(measurementDimension)
            obj = obj@g2o.core.BaseEdge(2, measurementDimension);
        end
    end
end
