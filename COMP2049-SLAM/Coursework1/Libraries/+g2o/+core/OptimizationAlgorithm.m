classdef OptimizationAlgorithm < handle
   
    properties(Access = protected)
       
        % The graph that we connect to
        optimizableGraph;
        
    end
    
    methods(Access = protected)
       
        function obj = OptimizationAlgorithm()
            
            
        end
        
    end
    
    methods(Access = public)
        
        function init(obj)
        end
        
        function [X, numberOfIterations] = solve(obj, X0, maximumNumberOfIterations)
            X=[];
            numberOfIterations = -1;
        end
        
    end
    
    
    methods(Access = public, Sealed)

        function optimizableGraph = graph(obj)
            optimizableGraph = obj.optimizableGraph;
        end
        
    end
    
    methods(Access =  {?g2o.core.SparseOptimizer})
        
        function setGraph(obj, optimizableGraph)
            % Check of the correct type
            assert(isa(optimizableGraph, 'g2o.core.OptimizableGraph') == true, ...
                'g2o:optimizationalgorithm:graphwrongtype', ...
                [ 'The graph should be of class g2o.OptimizableGraph' ...
                'the provided graph is of class %s'], class(optimizableGraph));
            
            obj.optimizableGraph = optimizableGraph;   
        end
        
    end
    
end