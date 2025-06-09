classdef BaseEdge < g2o.core.HyperGraphElement
    % BaseEdge summary of BaseEdge
    % An edge, also known as a factor, establishes a functional
    % relationship with or between one or more vertices (which store state
    % values). This functional relationship is typically related to the
    % unnormalized log probability of the process noise, observation noise,
    % or the initial condition.
    %
    % The functional relationship has a special form. We assume that every
    % edge is associated with a measurement. Measurement here means a
    % general way to input data into the system. It could be an actual
    % sensor measurement, or it could be used to convey a constraint.
    %
    % Given this measurement z, it's possible to construct an error
    % function which links the set of vertices {v} with the observation,    
    % 
    %     errZ = e({v}, z).
    %
    % When the error is Euclidean, this typically has the form
    %
    %     errZ = z - h({v})
    %
    % where h(.) is an observation model. However, an arbitrary error
    % function can be provided to handle nonlinear manifolds (typically
    % orientations). For example, when dealing with angles, angle wrapping
    % must be applied.
    %
    % The edge contributes the term
    %
    % chi2 = errZ' * Omega * errZ
    % 
    % to the cost. Note that Omega is the inverse covariance matrix of the
    % measurement.

    properties(Access = protected)
        
        % The measurement associated with this edge
        z;
        
        % The information associated with this edge
        Omega;
        
        % Dimensions of the measurement
        dimZ;
        
        % The number of vertices associated with this edge
        numVertices;
        
        % The error based on the current vertex state estimates
        errorZ;
        
        % The Jacobians
        J;

        % Cached
        H;
        b;
        
    end
    
    properties(Access = {?g2o.core.HyperGraphElement,?g2o.core.OptimizableGraph})
        % The vertices associated with this edge
        edgeVertices;
    end
    
    methods(Access = protected)
        
        function obj = BaseEdge(numVertices, measurementDimension)
            % BaseEdge Constructor for BaseEdge
            %
            % Syntax:
            %   obj = BaseEdge(numVertices, measurementDimension)
            %
            % Description:
            %   Creates an instance of a BaesEdge object. This is
            %   abstract and cannot be instantiated directly. Note that an
            %   edge is only valid when all the vertex "slots" have been
            %   assigned.
            %
            %   The method will also allocate internal error and
            %   information matrix structures, but these values are all set
            %   to NaNs and must be filled in.
            %
            % Inputs:
            %   numVertices - (int)
            %       The number of vertices this edge attaches to. Because
            %       this is a hypergraph, edges have to connect to at least
            %       one vertex, but can connect to many more.
            %
            %   measurementDimension - (int)
            %       The dimension of the measurement (z) within the vertex.
            %       The error function in the edge needs to produce a
            %       vector of this dimension. It does not relate at all to
            %       the dimensions of the individual vertices.
            %
            % Outputs:
            %   obj - (handle)
            %       An instance of a BaseEdge

            obj = obj@g2o.core.HyperGraphElement();
            
            assert(nargin > 1, 'g2o:baseedge:baseedge:insufficientarguments', ...
                'The number of vertices and dimensions are mandatory');
            
            % Check the number of vertices is good
           assert(numVertices > 0, 'g2o:baseedge:baseedge:invalidnumvertices', ...
                'The number of vertices must be a non-negative integer; the value is %d', numVertices);
            
            obj.numVertices = numVertices;
            
            obj.edgeVertices = cell(1, numVertices);
            
            % Check the dimensions are okay and store
            assert(measurementDimension > 0, 'g2o:baseedge:baseedge:invalidmeasurementdimension', ...
                'The measurement dimension must be a non-negative integer; the value is %d', measurementDimension);

            % Check the dimensions are okay and store
            obj.dimZ = measurementDimension;
            
            % Allocate an initial values
            obj.z = NaN(obj.dimZ, 1);
            obj.Omega = NaN(obj.dimZ, obj.dimZ);
            obj.errorZ = NaN(obj.dimZ, 1);
            
            obj.setId(g2o.core.BaseEdge.allocateId());
            
            % Preallocate space for the Jacobians for each vertex
            obj.J = cell(1, numVertices);


            obj.H = cell(obj.numVertices, obj.numVertices);
            obj.b = cell(1, obj.numVertices);

        end
    end
    
    methods(Access = public)
                
        function setInformation(obj, newOmega)
            % SETINFORMATION Set the information matrix for the edge.
            %
            % Syntax:
            %   obj.setInformation(newOmega);
            %
            % Description:
            %   Set the information matrix in the edge to the assigned
            %   value. This must be a postiive definite matrix whose
            %   dimension is the same as the measurement. Note that the
            %   information matrix is the inverse of the covariance.
            %
            %   The method checks that the information matrix has the
            %   correct dimensions and is positive definite.
            %
            % Inputs:
            %   newOmega - (nzxnz double)
            %       Positive definite information (inverse covariance)
            %       matrix.
            
            newOmegaNDims = ndims(newOmega);
            assert(newOmegaNDims == 2, ...
                'g2o:baseedge:setinformation:informationwrongdimension', ...
                'The information matrix must be a two dimensional array; ndims=%d', ...
                newOmegaNDims);
            
            % Get the vector sizes
            rows = size(newOmega, 1);
            cols = size(newOmega, 2);

            % Check the size
            assert((rows == obj.dimZ) && (cols == obj.dimZ), ...
                'g2o:baseedge:setinformation:informationwrongdimension', ...
                ['The information matrix should be a square ' ...
                ' matrix of dimension %d; the matrix has dimensions (%d, %d)'], ...
                obj.dimZ, rows, cols);
            
            % Check not NaN
            assert(any(any(isnan(newOmega))) == false, ...
                'g2o:baseedge:setinformation:informationhasnans', ...
                'The information matrix contains NaNs');
            
            % Check the matrix matrix is positive semidefinite
            assert(all(eig(newOmega) > eps), 'g2o:baseedge:setinformation:informationnotpsd', ...
                'The information matrix is not positive semidefinite');
            
            obj.Omega = newOmega;
        end

        
    end
    
    methods(Access = public)%, Sealed = true)
        
        function obj = setVertex(obj, vertexSlotNumber, vertex)
            % SETVERTEX Assign one of the slots of an edge to the vertex.
            %
            % Syntax:
            %   x = obj.setVertex(vertexSlotNumber, vertex)
            %
            % Description:
            %   Assign a vertex to one of the vertex slots. When you create
            %   an edge there are a number of "slots" which need to be
            %   filled which says which vertices the edge attaches to. This
            %   method handles that registration purpose. Validation
            %   checks that you are trying to post a vertex object to a
            %   valid slot number. A vertex can only appear in a given edge
            %   once. Internally, the method calls the vertex addEdge
            %   method so the vertex "knows" its associated with this edge.
            %
            %   If a slot already has a vertex in place, this method will
            %   override the vertex which was originally there.
            %
            % Inputs:
            %   vertexSlotNumber - (int)
            %       The slot number the vertex should be assigned to. This
            %       should be between 1 and numVertices
            %   vertex - (g2o.core.BaseVertex)
            %       The vertex to be slotted into the edge.
            
            % Cast to the right type
            vertexSlotNumber = uint32(vertexSlotNumber);
            
            % Check the vertex is of the right kind
            assert(isa(vertex, 'g2o.core.BaseVertex'), 'g2o:baseedge:addedge:wrongclass', ...
                'The object should inherit from base.Vertex; the object class is %s', ...
                class(vertex));
            
            % Check the vertex number if valid
            assert((vertexSlotNumber >=1) && (vertexSlotNumber <= obj.numVertices), ...
                'g2o:baseedge:addedge:invalidvertexnumber', ...
                'The vertex number must be between 0 and %d; the value is %d', obj.numVertices, ...
                vertexSlotNumber);
            
            % Assign
            obj.edgeVertices{vertexSlotNumber} = vertex;
            
            % (Re)allocate the storage of the Jacobian for this vertex
            obj.J{vertexSlotNumber} = zeros(vertex.dimension(), obj.dimZ);
            
            % Register the edge with the vertex
            vertex.addEdge(obj);
        end
        
        % Remove a vertex from this edge.
        function removeVertex(obj, vertex)
            % REMOVEVERTEX Remove the vertex from the edge.
            %
            % Syntax:
            %   obj.removeVertex(vertex);
            %
            % Description:
            %   Remove the vertex from the edge. The method works out which
            %   slot the vertex occupies and removes it. The vertex
            %   removeEdge method is called as well to update its
            %   book-keeping.
            %
            % Inputs:
            %   vertex - (g2o.core.BaseVertex)
            %       The vertex to be removed from the edge.
            
            % Check the class is correct
           assert(isa(vertex, 'g2o.core.BaseVertex'), 'g2o:baseedge:removeedge:wrongclass', ...
                'The object should inherit from base.Vertex; the object class is %s', ...
                class(vertex));
 
            % Go through all the registered vertices. If we found it,
            % remove it
            foundVertex = false;
            for v = 1 : obj.numVertices
                if ((isempty(obj.edgeVertices{v}) == false) && ...
                        (obj.edgeVertices{v}.id() == vertex.id()))
                    obj.edgeVertices{v}.removeEdge(obj);
                    obj.edgeVertices{v} = [];
                    foundVertex = true;
                    break;
                end
            end
            
            % The vertex wasn't found
            assert(foundVertex == true, 'g2o:baseedge:removeedge:wrongclass', ...
                'No vertex with ID %d is registered with edge with ID %d', vertex.id(), obj.elementId);
        end
        
        function numVertices = numberOfVertices(obj)
            % NUMBEROFVERTICES The number of vertex slots the edge has.
            %
            % Syntax:
            %   numVertices = obj.numberOfVertices();
            %
            % Description:
            %   Return the number of vertex slots the edge has. This
            %   specifies how many vertices the edge connects to.
            %
            % Outputs:
            %   numVertices - (int)
            %       The number of vertices the edge connects to.

            numVertices = obj.numVertices;
        end
        
        function numUndefinedVertices = numberOfUndefinedVertices(obj)
            % NUMBEROFUNDEFINEDVERTICES The number of vertex slots which have not been assigned.
            %
            % Syntax:
            %   numUndefinedVertices = obj.numberOfUndefinedVertices();
            %
            % Description:
            %   Returns how many vertex slots have not been assigned. For a
            %   valid edge, this must have a value of 0.
            %
            % Outputs:
            %   numUndefinedVertices - (int)
            %       The number of vertices which have not been defined it.

            numUndefinedVertices = sum(cellfun('isempty',obj.edgeVertices));
        end
        
        function v = vertex(obj, vertexSlotNumber)
            % VERTEX Return the vertex stored in the slot number.
            %
            % Syntax:
            %   vertex = obj.vertex(vertexSlotNumber);
            %
            % Description:
            %   Returns the vertex which has been registered in slot
            %   vertexSlotNumber. If the slot has not been assigned, an
            %   empty structure will be returned.
            %
            %   The method validates that the slot number lies in the range
            %   1 - numEdges.
            %
            % Outputs:
            %   vertex - (g2o.core.BaseVertex)
            %       The vertex instance stored in the slot.

            % Validate the slot number is good
            assert((vertexSlotNumber > 0) && (vertexSlotNumber <= obj.numVertices), ...
                'g2o:baseedge:illegalvertexid', 'The vertexNumber %d should be between 1 and %d', ...
                vertexSlotNumber, obj.numVertices);
            v = obj.edgeVertices{vertexSlotNumber};            
        end
        
        function vertices = vertices(obj)
            % VERTICES Return a cell array of the vertices stored in this edge.
            %
            % Syntax:
            %   vertices = obj.vertices();
            %
            % Description:
            %   Returns a cell array which contains all the vertices in
            %   this map. The cell array order is the same as the vertex
            %   slot number, and unassigned vertices will be empty entries
            %   in the cell array.
            %
            % Outputs:
            %   vertices - (cell array of g2o.core.BaseVertex)
            %       The cell array of the vertices.

           vertices = obj.edgeVertices;
        end
        
        function measurementDimension = rank(obj)
            % DIMENSION Returns the dimension of the measurement.
            %
            % Syntax:
            %   measurementDimension = obj.rank();
            %
            % Description:
            %   Returns the dimension of the measurement (error vector)
            %   computed by the edge.
            %
            % Outputs:
            %   measurementDimension - (int)
            %       The dimension of the measurement (error vector).

            measurementDimension = obj.dimZ;
        end
        
        function errZ = error(obj)
            % ERROR Returns the value of the computed error function.
            %
            % Syntax:
            %   errZ = obj.error();
            %
            % Description:
            %   Returns the output of the computed error function. As
            %   explained above, this is some measure of the difference
            %   between the measurement and the predicted measurement.
            %
            % Outputs:
            %   errZ - (measurementDimensionx1 double vector)
            %       The error vector.

            errZ = obj.errorZ;
        end
        
        function J = jacobianOplus(obj)
            % JACOBIANOPLUS Return the Jacobians of the error function.
            %
            % Syntax:
            %   J = obj.error();
            %
            % Description:
            %   The error function is, in general, a nonlinear function of
            %   the values of all the vertices attached with the edge. As
            %   part of the optimization process, the Jacobian of the error
            %   function with respect to all the vertices has to be
            %   computed.
            % 
            %   This method returns a cell array, sorted in vertex slot
            %   number, of the Jacobian of the error function with each
            %   vertex. The dimension of each Jacobian will be
            %   measurementDimension x vertexDimension.
            %
            % Outputs:
            %   J - (cell array of Jacobians)
            %       The cell array of the Jacobians

           J = obj.J;
        end
        
        function chi2 = chi2(obj)
            % CHI2 Compute and return the chi2 value of this edge.
            %
            % Syntax:
            %   chi2 = obj.chi2();
            %
            % Description:
            %   The cost returned by an edge is the real-valued scalar
            %   given by
            %
            %   chi2 = errZ' * omega * errZ
            %
            %   This method computes the value using a pre-computed value
            %   of errZ. 
            % 
            %   This method also checks that the computed chi2 value is
            %   not NaN.
            %
            %   The name comes from the fact that if errZ is Gaussian, the
            %   computed value has a chi2-distribution
            %
            % Outputs:
            %   chi2 - (double)
            %       The computed chi2 value.

            chi2 = obj.errorZ' * (obj.Omega * obj.errorZ);
            assert(isnan(chi2) == false, 'g2o:baseedge:chi2:chi2isnan', ...
                'The chi2 value is NaN.');
        end
        
        function [H, b] = computeHB(obj)
            % COMPUTEHB Compute the Hessian and b vector for this edge.
            %
            % Syntax:
            %   [H, b] = obj.computeHB();
            %
            % Description:
            %   Taking a Taylor series expansion of the error term leads to
            %   a quadratic in the error terms. The matrix H and vector b
            %   are intermediate quantities from this expansion. See
            %   section 2 of the paper "g2o: A general Framework for
            %   (Hyper) Graph Optimization" by Grisetti et al. for the
            %   definition.
            %
            %   The maths here is done fully automatically and does not
            %   require user involvement.
            %
            % Outputs:
            %   H - (measurementDimension x measurementDimension double)
            %       The Hessian matrix
            %   b - (measurementDimension x 1 double)
            %       The weighted error vector

            % Compute the error
            obj.computeError();

            % Compute the Jacobians
            obj.linearizeOplus();
            
            % Work out the contribution from each vertex. Note
            % this is symmetric, so we only populate the upper triangle.
            % We also cache common terms
            for i = 1 : obj.numVertices         
                JWi = obj.J{i}' * obj.Omega;
                obj.b{i} = JWi * obj.errorZ;
                obj.H{i, i} = JWi * obj.J{i};
                for j = i + 1 : obj.numVertices
                    obj.H{i, j} = JWi * obj.J{j};
                end
            end
            b = obj.b;
            H = obj.H;
        end
        
        function setMeasurement(obj, newZ)
            % SETMEASUREMENT Set the measurement in the vertex.
            %
            % Syntax:
            %   obj.setMeasurement(newZ);
            %
            % Description:
            %   Set the measurement for the edge. The measurement could,
            %   for example, be the value recorded from a GPS receiver.
            %
            %   The measurement is checked to have the correct number of
            %   dimensions and does not contain any NaN values.
            %
            % Inputs:
            %   newZ - (measurementDimension x 1 double)
            %       The observation associated with this edge.

            newZNDims = ndims(newZ);
            assert(newZNDims == 2, ...
                'g2o:baseedge:setmeasurement:measurementwrongdimension', ...
                'The measurement vector must be a column vector; ndims=%d', ...
                newZNDims);
            
            % Get the vector sizes
            rows = size(newZ, 1);
            cols = size(newZ, 2);

            % Check number of rows
            assert(cols == 1, ...
                'g2o:baseedge:setmeasurement:measurementwrongdimension', ...
                'The measurement vector must be a column vector; columns=%d', ...
                cols);
            
            % Check number of columns
            assert(rows == obj.dimZ, ...
                'g2o:baseedge:setmeasurement:measurementwrongdimension', ...
                'The measurement dimension is wrong; required=%d, actual=%d', ...
                obj.dimZ, rows);
            
            % Check not NaN
            assert(any(isnan(newZ)) == false, ...
                'g2o:baseedge:setmeasurement:measurementhasnans', ...
                'The measurement contains NaNs');
            obj.z = newZ;

        end
        
        function z = measurement(obj)
            % MEASUREMENT Return the measurement associated with the edge.
            %
            % Syntax:
            %   z = obj.measurement();
            %
            % Description:
            %   Return the measurement stored in the edge.
            %
            %   The measurement is checked to have the correct number of
            %   dimensions and does not contain any NaN values.
            %
            % Inputs:
            %   newZ - (measurementDimension x 1 double)
            %       The observation associated with this edge.

           z = obj.z; 
        end
        
        function Omega = information(obj)
            % INFORMATION Return the information matrix associated with the edge.
            %
            % Syntax:
            %   z = obj.information();
            %
            % Description:
            %   The information matrix is the inverse of the covariance
            %   matrix of the noise on the measurement. For a regular
            %   sensor observation, it is the observation covariance. For a
            %   prediction edge, it is the process noise.
            %
            % Outputs:
            %   information - (measurementDimension x measurementDimension double)
            %       The information matrix.
            Omega = obj.Omega;
        end

        function initialEstimate(obj)
            % INITIALESTIMATE Set the initial estimates of vertices.
            %
            % Syntax:
            %   obj.initialEstimate();
            %
            % Description:
            %   Some edges can construct initial estimates. This means, for
            %   example, that there is enough data in a vehicle prediction
            %   edge to predict what the next state will be. This can be
            %   used to initialize the state directly. This method would do
            %   this.
            %
            %   After the method is called the expectations is that some
            %   vertices will have their states updated.

            warning('BaseEdge.initialEstimate: not implemented');
        end           

        
       function validate(obj)
            % VALIDATE Validate that this edge instance is valid.
            %
            % Syntax:
            %   obj.validate();
            %
            % Description:
            %   A series of internal checks are run. The checks include
            %   that all the vertex slots are filled, the measurement is
            %   not NaN, and all the vertices are in the same graph (or not
            %   at all) as the edge. Generates an assertion on failure.

            if (obj.validated == true)
                return;
            end

            % If any edges are undefined, then generate a warning and
            % return
            assert(obj.numberOfUndefinedVertices() == 0, ...
                'g2o:baseedge:validate:undefinedvertices', ...
                'obj.numberOfUndefinedVertices()=%d for the edge with id %d', ...
                obj.numberOfUndefinedVertices(), obj.id);
            
            % Check the measurement is not nan
            assert(any(isnan(obj.z)) == false, ...
                'g2o:baseedge:validate:measurementhasnans', ...
                'The measurement contains NaNs');
            
            assert(any(any(isnan(obj.Omega))) == false, ...
                'g2o:baseedge:validate:informationhasnans', ...
                'The information matrix contains NaNs');

            assert(all(eig(obj.Omega) > eps), 'g2o:baseedge:validate:informationnotpsd', ...
                'The information matrix is not positive semidefinite');
            
            % Check all the vertices and make sure they are registered.
            for v = 1 : obj.numVertices
                assert(obj.edgeVertices{v}.owningGraph == obj.owningGraph, ...
                    'g2o:baseedge:validate:vertcesnotregistered', ...
                    'vertex with ID %d is not registered with a graph', ...
                    obj.edgeVertices{v}.elementId);
            end
            
            obj.validated = true;
        end    
        
    end
    
    methods(Access = public, Abstract)
        
        computeError(obj);
        
        
        linearizeOplus(obj);        
    end

    
     methods(Access = protected, Static)
       
        function id = allocateId()
            % ALLOCATEID Allocate a unique ID for this edge.
            %
            % Syntax:
            %   newID = g2o.core.BaseEdge.allocateId();
            %
            % Description:
            %   within the graph, some lookups are carried out based off of
            %   edge ID. This function returns a unique value for each
            %   edge instance. (It's actually given internally by
            %   incrementing a counter.)
            %
            % Outputs:
            %   newID - (int)
            %       A new ID for the edge.
            
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

