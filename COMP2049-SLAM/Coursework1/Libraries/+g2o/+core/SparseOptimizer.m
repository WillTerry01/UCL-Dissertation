% This class provides the full optimization support. The name is derived
% from the fact that it is designed to work with cost functions which have
% sparse patterns over variables. To use this class, you have to specify an
% optimization algorithm, which determines how the state variables are
% changed.

classdef SparseOptimizer < g2o.core.OptimizableGraph
    
    properties(Access = protected)
       
        % The optimization algorithm
        optimizationAlgorithm;

        % Use sparseinv?
        useSparseInv;
        
        % Estimated number of non-zero elements in the Hessian
        nonZeroElementsHXD;
        nonZeroElementsHXU;
        
        % Cached values
        HX;
        bX;
    end
    
    methods(Access = public)
       
        function obj = SparseOptimizer()
            
            obj = obj@g2o.core.OptimizableGraph();
            
            % If set to true, use the sparse inverse library
            obj.useSparseInv = (exist('sparseinv_mex', 'file') == 3);

            obj.optimizationReinitialized = true;
        end
    end
    
    methods(Access = public, Sealed)
        
        % This function makes it possible to over-ride the sparse
        % optimizer. Sometimes this can cause matlab to crash due to a mex
        % issue.        
        function useSparseInvIfItExists(obj, useSparseInv)
            if (exist('sparseinv_mex', 'file') == 3)
                obj.useSparseInv = useSparseInv;
            else
                obj.useSparseInv = false;
            end
        end
       
        % Set the optimization algorithm which is used
        function setAlgorithm(obj, optimizationAlgorithm)
            
            assert(isa(optimizationAlgorithm, 'g2o.core.OptimizationAlgorithm') == true, ...
                'g2o:sparseoptimizer:algorithmwrongclass', ...
                ['The algorithm should inhert from g2o.OptimizationAlgorithm.' ...
                'The class is %s'], class(optimizationAlgorithm));
            
            obj.optimizationAlgorithm = optimizationAlgorithm;
            
            optimizationAlgorithm.setGraph(obj);
        end
        
        % Return the optimization algorithm
        function algorithm = algorithm(obj)
            algorithm = obj.optimizationAlgorithm;
        end
        
        % This method computes the marginals for the specified verties. By
        % marginals, we mean the covariance values. If no vertices are
        % specified, the mean and covariance for the entire graph is
        % computed.
        function [X, PX] = computeMarginals(obj, vertices)
            
            % Check the graph is in a state where we can compute a value
            assert(obj.initializationRequired == false, ...
                'g2o:sparseoptimizer:initializationrequired', ...
                'Call initializeOptimization before extracting the marginals');

            % Flag if we want to skip the covariance
            getP = (nargout == 2);

            % Convenience processing - if no vertices are passed in, get
            % all of them; if a single vertex is passed in outside of a
            % cell array, wrap it in the cell array.
            if (nargin == 1)
                vertices = obj.verticesMap.values();
            elseif(iscell(vertices) == false)
                vertices = {vertices};
            end

            % How many vertices are there?
            numMarginalizedVertices = numel(vertices);

            % Allocate the mean and optionally the covariance
            X = cell(numMarginalizedVertices, 1);
            if (getP == true)
                PX = cell(numMarginalizedVertices, 1);
            end

            % Compute the inverse Hessian. This is needed if the
            % covariances are requested. Note that we can't compute this
            % if the Hessian is empty.

            % Use sparseinv if possible because it's generally faster.
            if ((getP == true) && (isempty(obj.HX) == false))
                if (obj.useSparseInv == true)
                    P = sparseinv(obj.HX);
                else 
                    P = inv(obj.HX);
                end
            end

            % Copy the estimates over; conditioned vertices have a
            % covariance of 0
            for v = 1 : numMarginalizedVertices
                X{v} = vertices{v}.estimate();
                if (getP == true)
                    if (vertices{v}.conditioned == true)
                        PX{v} = zeros(numel(X{v}), numel(X{v}));
                    else
                        PX{v} = P(vertices{v}.iX, vertices{v}.iX);
                    end
                end
            end
        end
        
        % This returns the computed information vector and information
        % matrix
        function [bX, HX] = computeHessian(obj)

            % Check the graph is in a state where we can compute a value
            assert(obj.initializationRequired == false, ...
                'g2o:sparseoptimizer:initializationrequired', ...
                'Call initializeOptimization before extracting the marginals');

            % Get the state and information matrix
            [HX, bX] = obj.computeHB(obj.X);
        end
    end
    
    methods(Access = protected)
        function numIterations = runOptimization(obj, maximumNumberOfIterations)
            
            assert(isempty(obj.optimizationAlgorithm) == false, ...
                'g2o:sparseoptimizer:optimizationalgorithmnotset', ...
                'The optimization algorithm is not set');
            
            [X, numIterations] = obj.optimizationAlgorithm.solve(obj.X, maximumNumberOfIterations, ...
                obj.optimizationReinitialized);

            obj.X = X;
        end

    end
    
    % These are various methods which only the optimization algorithm uses.
    % They are used to compute various quantities needed in Algorithm 1 of
    % "A Tutorial on Graph-Based SLAM" by Grisetti et al.
    
    methods(Access = {?g2o.core.OptimizationAlgorithm,?g2o.sampling.RiemannianHamiltonianSampler})
        
        function [HX,bX] = computeHB(obj, X)
            
            % Assign the state value to all the vertices. This is a bit
            % inefficient, but it simplifies the API because the edge only
            % needs to take the current vertex estimate.
            obj.assignXToVertices(X);
            
            n = length(obj.X);
            
            % Construct the Hessians used to compute the update. To speed
            % up construction, we compute the "upper triangle" and the
            % diagonal sub-blocks separately. We assemble them together
            % afterwards.
            % HXD = spalloc(n, n, obj.nonZeroElementsHXD);
            % HXU = spalloc(n, n, obj.nonZeroElementsHXU);

            % Construct the correction vector. This isn't sparse, so just
            % allocate it as a normal array.
            if (nargout == 2)
                obj.bX = zeros(n, 1);
            else
                obj.bX = NaN(n, 1);
            end
                        
            % Iterate over all the edges and get the H and b values for
            % each edge. Assemble them into HX and bX for the entire graph.
            % Note that HX is symmetric and only the upper right blocks are
            % filled in by each edge. Therefore, we only fill the upper
            % right triangle and then copy this over right at the end.
            % Note that the idx is empty for fixed vertices, so we skip
            % them.
            % edges = values(obj.edgesMap);
            % for e = 1 : length(edges)
            %     edge = edges{e};
            %     [H, b] = edge.computeHB();
            %     for i = 1 : length(edge.edgeVertices)
            %         idx = edge.edgeVertices{i}.iX;
            %         if (isempty(idx) == true)
            %             continue;
            %         end
            %         if (nargout == 2)
            %             obj.bX(idx) = obj.bX(idx) + b{i};
            %         end
            %         HXD(idx, idx) = HXD(idx, idx) + H{i, i};
            %         for j = i + 1 : length(edge.edgeVertices)
            %             jdx = edge.edgeVertices{j}.iX;
            %             HXU(idx, jdx) = HXU(idx, jdx) + H{i, j};
            %         end
            %     end
            % end
            % % Construct the full Hessian
            % obj.HX = HXU + HXU' + HXD;

            % Iterate over all the edges and get the H and b values for
            % each edge. Assemble them into HX and bX for the entire graph.
            % Note that HX is symmetric and only the upper right blocks are
            % filled in by each edge. Therefore, we only fill the upper
            % right triangle and then copy this over right at the end.
            % Note that the idx is empty for fixed vertices, so we skip
            % them.

            % Preallocate arrays for row indices, column indices, and values
            max_nnz = 1e6;  % Adjust based on your problem size
            row_indices = zeros(max_nnz, 1);
            col_indices = zeros(max_nnz, 1);
            mat_values = zeros(max_nnz, 1);
            
            % Counter for non-zero elements
            nnz_counter = 0;

            edges = values(obj.edgesMap);
            for e = 1 : length(edges)
                edge = edges{e};
                [H, b] = edge.computeHB();
                for i = 1 : length(edge.edgeVertices)
                    idx = edge.edgeVertices{i}.iX;
                    if (isempty(idx) == true)
                        continue;
                    end
                    if (nargout == 2)
                        obj.bX(idx) = obj.bX(idx) + b{i};
                    end

                    %HXD(idx, idx) = HXD(idx, idx) + H{i, i};
                
                    % Convert block to triplet form (find returns non-zero indices and values)
                    [local_row, local_col, local_vals] = find(H{i, i});
                
                    % Map local indices to global indices
                    row_indices(nnz_counter+1 : nnz_counter+numel(local_vals)) = idx(local_row);
                    col_indices(nnz_counter+1 : nnz_counter+numel(local_vals)) = idx(local_col);
                    mat_values(nnz_counter+1 : nnz_counter+numel(local_vals)) = local_vals;
                    
                    % Update counter
                    nnz_counter = nnz_counter + numel(local_vals);

                    for j = i + 1 : length(edge.edgeVertices)
                        jdx = edge.edgeVertices{j}.iX;

                        % Map local to global indices
                        [local_row, local_col, local_vals] = find(H{i, j});
                        row_idx = idx(local_row);
                        col_idx = jdx(local_col);
                        
                        % Accumulate both (i, j) and (j, i) for symmetry
                        row_indices(nnz_counter+1:nnz_counter+numel(local_vals)) = row_idx;
                        col_indices(nnz_counter+1:nnz_counter+numel(local_vals)) = col_idx;
                        mat_values(nnz_counter+1:nnz_counter+numel(local_vals)) = local_vals;
                        
                        % Add the symmetric entries
                        row_indices(nnz_counter+numel(local_vals)+1:nnz_counter+2*numel(local_vals)) = col_idx;
                        col_indices(nnz_counter+numel(local_vals)+1:nnz_counter+2*numel(local_vals)) = row_idx;
                        mat_values(nnz_counter+numel(local_vals)+1:nnz_counter+2*numel(local_vals)) = local_vals;
                    
                        % Update counter
                        nnz_counter = nnz_counter + 2*numel(local_vals);

                    end
                end
            end
            
            % Trim unused space
            row_indices = row_indices(1:nnz_counter);
            col_indices = col_indices(1:nnz_counter);
            mat_values = mat_values(1:nnz_counter);
            
            % Create the sparse matrix — this will sum any duplicate (row, col) entries automatically
            obj.HX = sparse(row_indices, col_indices, mat_values, numel(X), numel(X));

            %assert(norm(H2-obj.HX, 'fro') < 1e-6)

            if (nargout > 0)
                HX = obj.HX;
            end
            if (nargout > 1)
                bX = obj.bX;
            end
        end
                
        % Iterate through and set all the vertex states from the state
        % vector.
        function assignXToVertices(obj, X)
            
            % Check dimensions are the same
            assert(length(obj.X) == length(X), 'g2osparseoptimizer:assignxtovertices:xinconsistent', ...
                'The dimensions of X are inconsistent; length(obj.X)=%d, length(X)=%d', ...
                length(obj.X), length(X));

            % Iterate over all the vertices and assign the values to the
            % state vectors. This is not very efficient, but it means that
            % the state is set up correctly for each vertex.
            for v = 1 : obj.numNonFixedVertices
                obj.nonFixedVertices{v}.x = X(obj.nonFixedVertices{v}.iX);
            end
            
            % Update our local store of X
            obj.X = X;
        end
        
        % Iterate through and set all vertex states from the state vector
        % and the perturbed matrix. This uses the oplus operator on each
        % vertex. XdX is the global state vector updated with the new
        % values.
        function XdX = assignXToVerticesWithPerturbation(obj, X, dX)
            
            % Check dimensions are the same
            assert(length(obj.X) == length(X), 'g2osparseoptimizer:assignxtovertices:xinconsistent', ...
                'The dimensions of X are inconsistent; length(obj.X)=%d, length(X)=%d', ...
                length(obj.X), length(X));
            
            assert(length(obj.X) == length(dX), 'g2osparseoptimizer:assignxtovertices:xinconsistent', ...
                'The dimensions of X and dX are inconsistent; length(obj.X)=%d, length(dX)=%d', ...
                length(obj.X), length(dX));
            
            % Initialise the output vector to the right size
            XdX = zeros(size(X));
            
            % Iterate over all the vertices and assign the values to the
            % state vectors. This is not very efficient, but it means that
            % the oplus-added state values are available for the optimizer
            % to use in its next step.
            for v = 1 : obj.numNonFixedVertices
                obj.nonFixedVertices{v}.x = X(obj.nonFixedVertices{v}.iX);
                obj.nonFixedVertices{v}.oplus(dX(obj.nonFixedVertices{v}.iX));
                XdX(obj.nonFixedVertices{v}.iX) = obj.nonFixedVertices{v}.x;
            end
        end
    end
    
    methods(Access = protected)
        
        function buildStructure(obj)
            
            % Call the base class constructor
            buildStructure@g2o.core.OptimizableGraph(obj);
            
            % Now go through and estimate the number of non-zero elements
            % in the Hessian for preallocation of the sparse arrays.
            nzd = 0;
            nzu = 0;
            
            edges = values(obj.edgesMap);            
            for e = 1 : length(edges)
                edge = edges{e};
                for i = 1 : length(edge.edgeVertices)
                    idx = edge.edgeVertices{i}.iX;
                    if (isempty(idx) == true)
                        continue;
                    end
                    nzd = nzd + length(idx) * length(idx);
                    for j = i + 1 : length(edge.edgeVertices)
                        jdx = edge.edgeVertices{j}.iX;
                        nzu = nzu + length(idx) * length(jdx);
                    end
                end
            end
            obj.nonZeroElementsHXD = nzd;
            obj.nonZeroElementsHXU = nzu;
        end            
    end
    
end
