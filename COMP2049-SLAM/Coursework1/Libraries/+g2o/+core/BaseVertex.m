classdef BaseVertex < g2o.core.HyperGraphElement
    % BaseVertex summary of BaseVertex
    %
    % A base vertex stores a state value to be estimated. This could include
    % the pose (position and orientation of a robot), a landmark, or almost any
    % other type and kind of quantity of interest. The only requirement is that
    % the state can be described using a state vector.
    %
    % The class constructor is hidden. To create a vertex, you need to create a
    % subclass which specifies the dimension of the state. If the state vector
    % sits on a nonlinear manifold (e.g., it's SE(2)), then you should
    % overwrite the oplus method.
    %
    % Vertices can be "fixed" and their values do not change in the
    % optimization process. Fixed vertices are not represented in the
    % optimizer, which can provide both memory and computational advantages.

    properties(Access = protected)
        
        % The dimension of the state estimate
        dimX;
        
    end
    
    properties(Access = {?g2o.core.HyperGraphElement,?g2o.core.OptimizableGraph})
        % The indices in the big X array used in the optimizable graph
        % We let the optimizable graph access it directly to speed things
        % up a bit.
        iX;    
        
        % Specify if a vertex is fixed
        conditioned;
        
        % The state estimate
        x;
                        
        % The edges connected to this vertex
        edgesMap;
        
        % Cached for speed
        edgesArray;
        updateEdgesArray;
    end
    
    methods(Access = protected)
       
        function obj = BaseVertex(dimension)
            % BaseVertex Constructor for BaseVertex
            %
            % Syntax:
            %   obj = BaseVertex(dimension)
            %
            % Description:
            %   Creates an instance of a BaseVertex object. This is
            %   abstract and cannot be instantiated directly
            %
            % Inputs:
            %   dimension - (int)
            %       The dimension of the state stored in the vertex. For an
            %       (x,y) estimate it would be 2, for (x,y,theta) it would
            %       be 3, etc.
            %
            % Outputs:
            %   obj - (handle)
            %       An instance of a BaseVertex

            obj = obj@g2o.core.HyperGraphElement();
            
            assert(nargin > 0, 'g2o:basevertex:basevertex:insufficientarguments', ...
                'The dimension is mandatory');
            
            % Check the dimensions are okay and store
            assert(dimension > 0, 'g2o:basevertex::basevertex:dimensionwrong', ...
                'The dimension must be a non-negative integer; the value is %d', dimension);
            
            obj.dimX = dimension;
            
            % Allocate an initial value
            obj.x = NaN(obj.dimX, 1);
            
            % Not fixed by default
            obj.conditioned = false;
            
            % Automatically assign the Id
            obj.setId(g2o.core.BaseVertex.allocateId());
            
            % Preallocate
            obj.edgesMap = containers.Map('KeyType', 'int64', 'ValueType', 'any');
            
            obj.updateEdgesArray = true;
        end
    end
    
    methods(Access = public, Sealed = true)
        
        function dimension = dimension(obj)
            % DIMENSION The dimension of the state stored in the vertex
            %
            % Syntax:
            %   dimension = obj.dimension();
            %
            % Description:
            %   Returns the dimension of the state stored inside the base
            %   vertex.
            %
            % Outputs:
            %   dimension - (int)
            %       The dimension of the internally stored state.

            dimension = obj.dimX;
        end
        
        function edges = edges(obj)
            % EDGES Return the set of edges that attach to this vertex.
            %
            % Syntax:
            %   edges = obj.edges();
            %
            % Description:
            %   When the vertex is placed in a graph, it will be connected
            %   to other vertices through edges. This method returns the
            %   list of all edges that this vertex is connected to.
            %
            % Outputs:
            %   edges - (cell array of g2o.core.BaseEdge)
            %       A cell array which contains all the edges this vertex
            %       is connected to.
            
            if (obj.updateEdgesArray == true)
                obj.edgesArray = values(obj.edgesMap);
                obj.updateEdgesArray = false;
            end
            edges = obj.edgesArray;
        end
        
        function numEdges = numberOfEdges(obj)
            % NUMBEROFEDGES Return the number of edges this vertex connects to.
            %
            % Syntax:
            %   numEdges = obj.numberOfEdges();
            %
            % Description:
            %   This returns the number of edges this vertex is connected
            %   to. It is the length of the cell array returned by the
            %   edges() method.
            %
            % Outputs:
            %   numEdges - (int)
            %       The number of edges this vertex.
            % 
            % See also:
            %   EDGES

            numEdges = length(obj.edgesMap);
        end
        
        function estimate = estimate(obj)
            % ESTIMATE Return the estimated value stored in this vertex
            %
            % Syntax:
            %   estimate = obj.estimate();
            %
            % Description:
            %   As noted in the class description, each vertex stores a
            %   state estimate which is a vector of known dimension. This
            %   method returns the value of this state vector
            %
            % Outputs:
            %   estimate - (dx1 vector)
            %       The value of the etimated state 
            % 
            % See also:
            %   DIMENSION

            estimate = obj.x;
        end
        
        function isFixed = fixed(obj)
            % FIXED Is the state in the vertex fixed?
            %
            % Syntax:
            %   isFixed = obj.fixed();
            %
            % Description:
            %   Although vertices normally contain states whose values are
            %   to be estimated, we can pin the values of some vertices so
            %   their values do not change. This makes it possible to, for
            %   example, freeze most of a system and only update a small
            %   part of it.
            %
            %   A vertex which is fixed is not assigned memory in the
            %   optimizer. The hessianIndex() is null.
            %
            %   Since a fixed vertex is mathematically equivalent to
            %   conditioning upon its value, fixed vertices are assigned a
            %   covariance of 0 when computing marginals
            %
            % Outputs:
            %   isFixed - (bool)
            %       Flag indicates if the vertex is fixed.
            % 
            % See also:
            %   SETFIXED, HESSIANINDEX
            
            isFixed = obj.conditioned;
        end
        
        % Set whether the vertex is fixed or not.
        function setFixed(obj, fixed)
            % SETFIXED Set the fixed state of a vertex.
            %
            % Syntax:
            %   obj.setFixed(fixed);
            %
            % Description:
            %   Set whether a vertex is fixed or not. See the fixed()
            %   method for more details.
            %
            % Input:
            %   fixed - (bool)
            %       Flag to indicate if the vertex is fixed.
            % 
            % See also:
            %   FIXED, HESSIANINDEX

            obj.conditioned = fixed;
        end
        
        function xIndices = hessianIndex(obj)
            % HESSIANINDEX The index of where the vertex state maps to the
            % underlying hessian
            %
            % Syntax:
            %   xIndices = obj.hessianIndex();
            %
            % Description:
            %   Internally, the graph assembles all the states of all the
            %   individual vertices into a giant centralized database. This
            %   index is used to construction the system Hessian, hence the
            %   name. Fixed vertices or vertices not registered with a map
            %   will return an empty array.
            %
            % Output:
            %   xIndices - (dx1 array of ints)
            %       A continguous set of ints of length d which map to the
            %       local of the vertex in the Hessian.

            xIndices = obj.iX;
        end
        

        function setEstimate(obj, newX)
            % SETESTIMATE Set the estimate in the vertex.
            %
            % Syntax:
            %   obj.setEstimate(newX);
            %
            % Description:
            %   Set the state of the vertex to the value stored in the
            %   array x. The dimension needs to be the same that 
            %
            % Input:
            %   newX - (dx1 double array)
            %       The state of the vertex.
      
            % Needs to be a 2D array
            newXNDims = ndims(newX);
            assert(newXNDims == 2, ...
                'g2o:basevertex:setestimate:estimatewrongdimension', ...
                'The estimate vector must be a column vector; ndims=%d', ...
                newXNDims);
            
            % Get the vector sizes
            rows = size(newX, 1);
            cols = size(newX, 2);

            % Check number of rows
            assert(cols == 1, ...
                'g2o:basevertex:setestimate:estimatewrongdimension', ...
                'The estimate vector must be a column vector; columns=%d', ...
                cols);
            
            % Check number of columns
            assert(rows == obj.dimension, ...
                'g2o:basevertex:setestimate:estimatewrongdimension', ...
                'The estimate dimension is wrong; required=%d, actual=%d', ...
                obj.dimX, rows);
            
            % Check not NaN
            assert(any(isnan(newX)) == false, ...
                'g2o:basevertex:setestimate:estimatehasnans', ...
                'The estimate contains NaNs');

            obj.x = newX;
        end
        
        function validate(obj)
            % VALIDATE Validate that this vertex instance is valid.
            %
            % Syntax:
            %   obj.validate();
            %
            % Description:
            %   A series of internal checks are run to make sure that the
            %   state vector does not contain any NaN values, and all the
            %   edges the vertex is attached to are registered in the same
            %   graph (or no graph at all).
            
            % If already validated, return
            if (obj.validated == true)
                return
            end
            
            % Check if any NaNs are there
            % Check the dimension; this should be superfluous
            assert(any(isnan(obj.x)) == false, ...
                'g2o:basevertex:validate:estimatehasnans', ...
                'The estimate contains NaNs');
            
            % Check all the edges and make sure they are registered
            edges = values(obj.edgesMap);
            for e = 1 : length(edges)
                assert(edges{e}.owningGraph == obj.owningGraph, ...
                    'g2o:basevertex:validate:edgenotregistered', ...
                    'edge with ID % is not registered with a graph', obj.edges{e}.elementId);
            end            
            obj.validated = true;
        end
        
    end
    
    methods(Access = public)

        function oplus(obj, update)
            % OPLUS Apply an incremental update to the state estimate.
            %
            % Syntax:
            %   obj.oplus(update);
            %
            % Description:
            %   The output of each optimisation step is a small increment
            %   to the vertex estimate. This method applies this correction
            %   step. The actual applications depends on the nature of the
            %   stored quantity. For Euclidean quantities (e.g., x,y) this
            %   is simply an add operation. For nonlinear manifolds (most
            %   typically orientations) you need to take account of that.
            %   The default implementation adds update directly to the
            %   state vector.
            %
            % Inputs:
            %   update - (dx1 double)
            %       Small perturbed update to the state estimate. This will
            %       have the same state dimension as the vertex state.
            
            obj.x = obj.x + update;
        end
        
        function setToOrigin(obj)
            % SETTOORIGIN Set the vertex state to its "origin" or "zero"
            % value.
            %
            % Syntax:
            %   obj.setToOrigin();
            %
            % Description:
            %   Set the vertex state to a reset or default value. For a
            %   Euclidean space, this is normally the zero vector. However,
            %   this isn't always the case. e.g., for quaternions it might
            %   be [0,0,0,1]'.

            obj.x = zeros(obj.dimX, 1);
        end
    end
    
    methods(Access = {?g2o.core.HyperGraph,?g2o.core.BaseEdge}, Sealed = true)
        
        function obj = addEdge(obj, edge)
            % ADDEDGE Update the internal edge record to add a new edge.
            %
            % Syntax:
            %   obj.addEdge(edge);
            %
            % Description:
            %   When a vertex is added to an edge, the list within the
            %   vertex, which stores the edges it is associated with, has
            %   to be updated. This method undertakes this book-keeping
            %   operation.
            %
            % Inputs:
            %   edge - (g2o.core.BaseEdge)
            %       The new edge that the vertex is being attached to.

            obj.edgesMap(edge.id) = edge;
            obj.updateEdgesArray = true;
        end
        
        function obj = removeEdge(obj, edge)
            % REMOVEDGE Update the internal edge record to remove an edge.
            %
            % Syntax:
            %   obj.removeEdge(edge);
            %
            % Description:
            %   When a vertex is removed from an edge, the list within the
            %   vertex, which stores the edges it is associated with, has
            %   to be updated. This method undertakes this book-keeping
            %   operation.
            %
            %   This method will throw an assert if you remove an edge
            %   which is not present.
            %
            % Inputs:
            %   edge - (g2o.core.BaseEdge)
            %       The edge being removed from the vertex.

            assert(isKey(obj.edgesMap, edge.id) == true, 'g2o:basevertex:removeedge:repeatid', ...
                'Attempt to remove unregistered edge %d', edge.id);
            remove(obj.edgesMap, edge.id);
            obj.updateEdgesArray = true;
        end
    end
    
    methods(Access = {?g2o.core.OptimizableGraph}, Sealed = true)
        
        function setXIndices(obj, xIndices)
            % SETXINDICES Set the Hessian Index for the vertex.
            %
            % Syntax:
            %   obj.setXIndices(xIndices);
            %
            % Description:
            %   As explained in hessianIndex(), internally, the graph assembles
            %   all the states of all the individual vertices into a giant
            %   centralized state. This internal method is called to set
            %   what the index is.
            %
            % Inputs:
            %   xIndices - (dx1 array of ints)
            %       A continguous set of ints of length d which map to the
            %       local of the vertex in the Hessian.

            obj.iX = xIndices;
        end        
    end
    
           
     methods(Access = protected, Static)
       
        function id = allocateId()
            % ALLOCATEID Allocate a unique ID for this vertex.
            %
            % Syntax:
            %   newID = g2o.core.BaseVertex.allocateId();
            %
            % Description:
            %   within the graph, some lookups are carried out based off of
            %   vertex ID. This function returns a unique value for each
            %   vertex instance. (It's actually given internally by
            %   incrementing a counter.)
            %
            % Outputs:
            %   newID - (int)
            %       A new ID for the vertex.
            
            persistent idCount;
            
            if (isempty(idCount))
                idCount = 0;
            else
                idCount = idCount + 1;
                
            end
            id = idCount;
        end
     end
end