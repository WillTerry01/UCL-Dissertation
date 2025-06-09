classdef TestUnaryEdge < g2o.BaseUnaryEdge
    
    methods(Access = public)
   
        function obj = TestUnaryEdge()
            obj = obj@g2o.BaseUnaryEdge(2);
        end
        
    end
    
end