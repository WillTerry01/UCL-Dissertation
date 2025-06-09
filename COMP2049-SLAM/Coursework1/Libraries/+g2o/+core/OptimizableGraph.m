% This class provides an interface ontop of the basic hypergraph to support
% setting up and optimizing the graph. It does not, however, actually
% implement any optimization algorithm. This is an internal class and
% shouldn't be instantiated directly.

classdef OptimizableGraph < g2o.core.HyperGraph
    
    properties(Access = protected)
        
        % Flag shows if initialization is required
        initializationRequired;
        
        % Flag shows if the optimizer has been reinitialized since the last
        % solve call
        optimizationReinitialized;

        % The state vector
        X;
        
        % The non-fixed vertices
        numNonFixedVertices;
        nonFixedVertices;       
    end
    
    methods(Access = public)

        function obj = OptimizableGraph()
            
            obj = obj@g2o.core.HyperGraph();
            
            obj.initializationRequired = false;
            
        end
        
        function obj = addVertex(obj, vertex)
            
            % Call the method in the base class
            addVertex@g2o.core.HyperGraph(obj, vertex);
            
            % Flag initialization required
            obj.initializationRequired = true;
            
        end

        function initializationRequired = isInitializationRequired(obj)
            initializationRequired = obj.initializationRequired;
        end
 
        function obj = removeVertex(obj, vertex)
            
            % Call the method in the base class
            removeVertex@g2o.core.HyperGraph(obj, vertex);
            
            % Flag initialization required
            obj.initializationRequired = true;
            
        end
        
        function obj = addEdge(obj, edge)
            
            % Call the method in the base class
            addEdge@g2o.core.HyperGraph(obj, edge);
            
            % Flag initialization required
            obj.initializationRequired = true;
            
        end
 
        function obj = removeEdge(obj, edge)
            
            % Call the method in the base class
            removeEdge@g2o.core.HyperGraph(obj, edge);
            
            % Flag initialization required
            obj.initializationRequired = true;
            
        end

        function obj = initializeOptimization(obj, validateGraph)
            if ((nargin == 2) && (validateGraph == true))
                obj.validateGraph();
            end
            obj.buildStructure();
            obj.computeInitialErrors();
            obj.initializationRequired = false;
            obj.optimizationReinitialized = true;
        end
        
        function numIterations = optimize(obj, maximumNumberOfIterations)
            
            assert(obj.initializationRequired == false, ...
                'g2o:optimizablegraph:initializationrequired', ...
                'The graph structure has changed; call initializeOptimization before optimizing');
            
            % Default number of iterations
            if (nargin == 1)
                maximumNumberOfIterations = 50;
            end

            numIterations = obj.runOptimization(maximumNumberOfIterations);

            % Now clear the reinitialized flag
            obj.optimizationReinitialized = false;
        end
    
        % Get the chi2 value for the overall graph. This is the cost.
        % If specified, also return a list of each chi2 value for each edge
        % separately.
        function [chi2Sum, chi2List] = chi2(obj)
            
            chi2Sum = 0;

            edgesToCheck = values(obj.edgesMap);
            numberOfEdges = length(edgesToCheck);

            if (nargout == 2)
                chi2List = zeros(1, numberOfEdges);
            end
            
            for e = 1 : length(edgesToCheck)
                edgesToCheck{e}.computeError();
                chi2Value = edgesToCheck{e}.chi2();
                chi2Sum = chi2Sum + chi2Value;
                if (nargout == 2)
                    chi2List(e) = chi2Value;
                end                
            end            
        end

        function X = getX(obj)
            X = obj.X;
        end
    end
    
    methods(Access = protected)
       
        % Make sure that all the vertices and edges are set up correctly
        function validateGraph(obj)
            
            % Make sure the progress bar is reset
            textprogressbar();

            % The sets to check
            verticesToCheck = values(obj.verticesMap);
            edgesToCheck = values(obj.edgesMap);
            
            % First clear the valid state on all vertices and edges
            for v = 1 : length(verticesToCheck)
                verticesToCheck{v}.clearValidated();
            end
            for e = 1 : length(edgesToCheck)
                edgesToCheck{e}.clearValidated();
            end
            
            % Check all the vertices are valid
            textprogressbar('validating vertices: ');
            for v = 1 : length(verticesToCheck)
                textprogressbar(100*v/length(verticesToCheck));
                verticesToCheck{v}.validate();
            end
            textprogressbar(' completed');

            % Check all the edges are valid
            textprogressbar('validating edges   : ');
            for e = 1 : length(edgesToCheck)
                textprogressbar(100*e/length(edgesToCheck));
                edgesToCheck{e}.validate();
            end
            textprogressbar(' completed');
        end
        
        % Internally process the structure and pre-cache results where
        % possible.
        function buildStructure(obj)
            
            % First accumulate all the states into a big vector
            currentIdx = 0;
            
            obj.X = [];
            
            % Go through each vertex. Identify the non-fixed ones. Store
            % them in the big X matrix and assign the index in the big X
            % map
            verticesToCheck = values(obj.verticesMap);
            obj.nonFixedVertices = cell(length(verticesToCheck), 1);
            obj.numNonFixedVertices = 0;
            for v = 1 : length(verticesToCheck)
                vertex = verticesToCheck{v};
                if (vertex.conditioned == true)
                    vertex.iX = [];
                    continue
                end
                vertexDim = vertex.dimension();
                idx = currentIdx + (1:vertexDim);
                currentIdx = idx(end);
                vertex.iX = idx;
                obj.X(idx, 1) = vertex.estimate();
                obj.numNonFixedVertices = obj.numNonFixedVertices + 1;
                obj.nonFixedVertices{obj.numNonFixedVertices} = vertex;
            end
        end
        
        function computeInitialErrors(obj)
            edgesToCheck = values(obj.edgesMap);            
            for e = 1 : length(edgesToCheck)
                edgesToCheck{e}.computeError();
            end
        end
    end
    
    methods(Access = protected, Abstract)
        numberOfIterations = runOptimization(obj, maximumNumberOfIterations);
    end
    
end