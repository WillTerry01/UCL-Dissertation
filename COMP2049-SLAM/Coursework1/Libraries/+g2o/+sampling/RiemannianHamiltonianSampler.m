classdef RiemannianHamiltonianSampler < g2o.sampling.HamiltonianSampler
   
    % This applies the RHMC sampler to a graph. The main assumption is that
    % the information matrices on each edge do not depend on the state.
    
    properties(Access = protected)
        
        % The graph we are sampling from
        graph;
        
    end
    
    methods(Access = public)
       
        function obj = RiemannianHamiltonianSampler(graph)
           
            % Check the graph is of the right kind
            assert(isa(graph, 'g2o.core.SparseOptimizer'), 'graphbasedsampler:graphbasedsampler:wrongclass', ...
                'The object should be of class g2o.core.SparseOptimizer; the object class is %s', ...
                class(graph));
            
            % Run base constructor
            obj = obj@g2o.sampling.HamiltonianSampler();
            
            obj.graph = graph;
            
        end
        
    end
    
    methods(Access = protected)
       
        % Compute the likelihood from the graph
        function V = computeV(obj)
            obj.graph.assignXToVertices(obj.theta);
            V = 0.5 * obj.graph.chi2();
        end
        
        % Compute the mass matrix. This is identical to the Hessian.
        function computeM(obj)
            obj.graph.assignXToVertices(obj.theta);
            [~, obj.M] =  obj.graph.computeHessian();
        end
        
        function dHDTheta = computeDHDTheta(obj)
            obj.graph.assignXToVertices(obj.theta);
            [dHDTheta, ~] = obj.graph.computeHessian();
        end        
    end
end