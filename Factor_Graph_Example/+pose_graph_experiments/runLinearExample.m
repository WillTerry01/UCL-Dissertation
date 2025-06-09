function [chi2, chi2List, X, Px, dimX, dimZ] = ...
    runLinearExample(numberOfTimeSteps, omegaRScale, omegaQScale, ...
    testProposition4, numObs, obsPeriods, numSubgraph, scenario, R, Q)

    import g2o.core.*;
    import pose_graph_experiments.*;
    import two_d_tracking_model_answer.*;

    % Check if obsPeriods is provided, if not, set it to a default value
    if (nargin < 6 || isempty(obsPeriods))
        obsPeriods = ones(1, numSubgraph);
    end

    % Check if numSubgraph is provided, if not, set it to 1
    if (nargin < 7)
        numSubgraph = 1;
    end

    % Check which scenario, default Scenario 1
    if nargin < 8
        scenario = 1;
    end

    % Some parameters
    dT = 1;
    sigmaR = 1;
    sigmaQ = 1;

    % Work out the state transition equations for the simulator
    F0=[1 dT; 0 1];
    Q0=[dT^3/3 dT^2/2;dT^2/2 dT] * sigmaQ;

    F = [F0 zeros(2); zeros(2) F0];

    % If R is not provided, use default
    if nargin < 9
        R = eye(2) * sigmaR;
    end

    % If Q is not provided, use default
    if nargin < 10
        Q = [Q0 zeros(2); zeros(2) Q0];
    end

    H = [1 0 0 0;
        0 0 1 0];

    % Work out the information matrices
    omegaR = omegaRScale * inv(R);
    omegaQ = omegaQScale * inv(Q);

    % Ground truth array
    trueX = zeros(4, numberOfTimeSteps);
    z = zeros(2, numberOfTimeSteps);

    % First timestep
    trueX(2, 1) = 0.1;
    trueX(4, 1) = -0.1;
    z(:, 1) = H * trueX(:, 1) + sqrtm(sigmaR) * randn(2, 1);

    % Now predict the subsequent steps
    for k = 2 : numberOfTimeSteps
        trueX(:, k) = F * trueX(:, k - 1) + sqrtm(Q) * randn(4, 1);
        z(:, k) = H * trueX(:, k) + sqrtm(sigmaR) * randn(2, 1);
    end

    timeStepsPerSubgraph = floor(numberOfTimeSteps / numSubgraph);
    remainder = mod(numberOfTimeSteps, numSubgraph);

    chi2 = 0;
    chi2List = [];

    % Initialise dimX and dimZ arrays
    dimX = zeros(1, numSubgraph);
    dimZ = zeros(1, numSubgraph);

    % Initialise mean and covariance cell for each subgraph
    X = cell(1, numSubgraph);
    Px = cell(1, numSubgraph);

    for subgraphIndex = 1 : numSubgraph
        % Adjust the number of time steps for the last subgraph
        if subgraphIndex == numSubgraph
            timeStepsPerSubgraph = timeStepsPerSubgraph + remainder;
        end

        % Determine current observation period
        currentObsPeriod = obsPeriods(subgraphIndex);

        % Create the graph
        graph = SparseOptimizer();
        algorithm = GaussNewtonOptimizationAlgorithm();
        graph.setAlgorithm(algorithm);

        % This array contains the set of vertices for the target state over time
        v = cell(timeStepsPerSubgraph, 1);

        % This keeps count of the vertex number
        idx = 1;

        % This is the last timestep when a vertex was created
        lastN = 0;

        if scenario == 1
            % Scenario 1: Constant time intervals
            for n = 1 : timeStepsPerSubgraph  
                takeObservation = rem((subgraphIndex-1)*timeStepsPerSubgraph + n, currentObsPeriod) == 0;
                % Create the vertex for each time step
                addVertexWithObservation(n, takeObservation);
            end
        elseif scenario == 2
            % Scenario 2: Vertices only when observations are taken
            for n = 1 : timeStepsPerSubgraph
                takeObservation = (n==1) || ((subgraphIndex-1)*timeStepsPerSubgraph + n < numObs || rem((subgraphIndex-1)*timeStepsPerSubgraph + n, currentObsPeriod) == 0);
                if takeObservation
                    addVertexWithObservation(n, true);
                end
            end
        end

        % Graph construction complete
        
        % Initialise the optimization. This is done here because it's a bit
        % expensive and if we cache it, we can do it multiple times
        graph.initializeOptimization();
        
        % Optimize
        if (testProposition4 == true)
            graph.optimize();
        end
        
        % Compute the chi2 value
        [chi2_subgraph, ~] = graph.chi2();
        
        % For subgraph case, the chi2 stores summed chi2 value for each
        % subgraphs. chi2List stores the corresponding chi2 value for each
        % subgraph.
        chi2 = chi2 + chi2_subgraph;
        chi2List = [chi2List; chi2_subgraph];

        % If all outputs needed
        if (nargout == 6)
            edges = graph.edges();
            vertices = graph.vertices();
        
            % Get the covariance of vertices
            [X{subgraphIndex}, Px{subgraphIndex}] = graph.computeMarginals();
        
            for ed = 1 : length(edges)
                dimZ(subgraphIndex) = dimZ(subgraphIndex) + edges{ed}.dimension();
            end
        
            for ve =  1 : length(vertices)
                dimX(subgraphIndex) = dimX(subgraphIndex) + vertices{ve}.dimension();
            end
        end
    end

    function addVertexWithObservation(n, takeObservation)
            % Create the new vertex object
            v{idx} = ObjectStateVertex();
    
            original_timestep = timeStepsPerSubgraph - remainder;
            % Set the initial estimate.
            if (testProposition4 == false)
                v{idx}.setEstimate(trueX(:, (subgraphIndex-1)*original_timestep + n));
            else
                v{idx}.setEstimate(0*trueX(:, (subgraphIndex-1)*original_timestep + n));
            end
    
            % Add the vertex to the graph.
            graph.addVertex(v{idx});
    
            % If this isn't the first vertex, add the dynamics. The
            % prediction interval is a function of the time since the last
            % vertex was created
            if (idx > 1)
                processModelEdge = ObjectProcessModelEdge();
                processModelEdge.setVertex(1, v{idx-1});
                processModelEdge.setVertex(2, v{idx});
                processModelEdge.setMeasurement([0;0;0;0]);
    
                dTp = dT * (n - lastN);
    
                F1=[1 dTp; 0 1];
                Q1=[dTp^3/3 dTp^2/2;dTp^2/2 dTp] * sigmaQ;
    
                processModelEdge.setF(blkdiag(F1, F1));
                processModelEdge.setInformation(omegaQScale * inv(blkdiag(Q1, Q1)));
                graph.addEdge(processModelEdge);
            end
    
            if takeObservation
                % Create the measurement edge
                e = ObjectMeasurementEdge();
    
                % Link it so that it connects to the vertex we want to estimate
                e.setVertex(1, v{idx});
    
                % Set the measurement value and the measurement covariance
                e.setMeasurement(z(:,(subgraphIndex-1)*original_timestep + n));
                e.setInformation(omegaR);
    
                % Add the edge to the graph; the graph now knows we have these edges
                % which need to be added
                graph.addEdge(e);
            end
    
            % Bump the current vertex index
            idx = idx + 1;
    
            % Store the time step the observation was taken
            lastN = n;
        end
end

