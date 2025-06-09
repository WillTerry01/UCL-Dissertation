classdef TestVertex < g2o.BaseVertex

    methods(Access = public)
        
        function obj = TestVertex()
            obj = obj@g2o.BaseVertex(2);
        end
        
    end
    
end
