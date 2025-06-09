% THIS SCRIPT IS USED TO TEST 2D LINEAR EXAMPLE. PLEASE ONLY COMMENT LINES
% ABOVE 121 AND CHANGE THE FUNCTION NAME TO "linear2Dof", IF WE ARE
% TRYING TO TEST BO ON 2D CASE. PLEASE UNCOMMENT LINES 1- 121 AND CHANGE
% THE FUNCTION NAME TO "linear2dof" IF TESTING THE MEAN AND COVARIANCE
% VALUES.

% Number of steps per episode
numberOfTimeSteps = 200;
numberOfEpisodes = 1000;

scenario = 1;
omegaRScale = 1;
omegaQScale = 1;
testProposition4 = false;
numObs = 50;
obsPeriods = [1 5 10];
% obsPeriods = 1;
numSubgraph = length(obsPeriods);

% R11s = 0.1:0.2:2;
% R22s = 0.1:0.2:2;
% Q11s = 1:0.2:3;
% Q22s = 1:0.2:3;

% GROUND TRUTH
R11s = 0.2;
R22s = 0.1;
Q11s = 1;
Q22s = 2;

totalCombinations = length(R11s) * length(R22s) * length(Q11s) * length(Q22s);

% Flatten the grid into a single list of combinations
allCombinations = combvec(R11s, R22s, Q11s, Q22s)';

for idx = 1:totalCombinations
    R11 = allCombinations(idx, 1);
    R22 = allCombinations(idx, 2);
    Q11 = allCombinations(idx, 3);
    Q22 = allCombinations(idx, 4);

    % Create sigmaR and sigmaQ for this combination
    sigmaR = diag([R11, R22]);
    sigmaQ = diag([Q11, Q22]);

    % Initialize episode-specific storage
    if numSubgraph == 1
        % Compute the number of edges
        numberOfEdges = 2 * numObs - 1 + ...
            floor((numberOfTimeSteps - numObs) / obsPeriods) + ...
            numberOfTimeSteps - numObs;
        chi2Store = zeros(numberOfEpisodes, numberOfEdges);
        chi2SumStore = zeros(numberOfEpisodes, 1);
    else
        chi2Store = zeros(numberOfEpisodes, numSubgraph);
        chi2SumStore = zeros(numSubgraph, 1);
    end


    % Call the function
    [chi2SumStore(1), chi2Store(1, :), ~, ~, dimX, dimZ] = linear2dof(numberOfTimeSteps, omegaRScale, omegaQScale, ...
        testProposition4, numObs, obsPeriods, numSubgraph, scenario, sigmaR, sigmaQ); % <- Use your specific input arguments here

    parfor r = 2 : numberOfEpisodes
        fprintf('%03d\n', r)
        [chi2SumStore(r), chi2Store(r, :)] = ...
            linear2dof(numberOfTimeSteps, ...
            omegaRScale, omegaQScale, testProposition4, ...
            numObs, obsPeriods, numSubgraph, scenario, sigmaR, sigmaQ);
    end


    if numSubgraph == 1
        % Compute the number of degrees of freedom
        if (testProposition4 == true)
            N = dimZ - dimX;
        else
            N = dimZ;
        end
        meanChi2 = mean(chi2SumStore);
        covChi2 = cov(chi2SumStore);

        figure(1)
        plot(chi2SumStore)
        title(sprintf('Mean: %f; Covariance %f', meanChi2, covChi2))
        
        C = abs(log(meanChi2/N)) + abs(log(covChi2/(2*N)));
        fid = fopen('D:\University\UCL\project\2dlinearResults_obs1.txt', 'a+');
        str = strjoin({sprintf('%.2f %.2f %.2f %.2f', R11, R22, Q11, Q22), sprintf(' %d', C), sprintf(' %d\n', sum(C))});
        fprintf(fid, '%s', str);
        fclose(fid);
    else

        % Compute the mean and covariance for each subgraph
        C = zeros(1, numSubgraph);
        for i = 1:numSubgraph
            % Compute the number of degrees of freedom
            if (testProposition4 == true)
                N = dimZ(i) - dimX(i);
            else
                N = dimZ(i);
            end

            figure(i)
            plot(chi2Store(:, i))
            meanChi2 = mean(chi2Store(:, i));
            covChi2 = cov(chi2Store(:, i));
            title(sprintf('Subgraph: %d, Correct N: %d, Mean: %f; Covariance %f', i, N, meanChi2, covChi2))
    
            % Compute the Consistency Measurement for each subgraph
            C(i) = abs(log(meanChi2/N)) + abs(log(covChi2/(2*N)));
        end
        C_Sum = sum(C);

        fid = fopen('D:\University\UCL\project\2dlinearResults_obs1_5_10.txt', 'a+');
        str = strjoin({sprintf('%.2f %.2f %.2f %.2f', R11, R22, Q11, Q22), sprintf(' %d', C), sprintf(' %d\n', sum(C))});
        fprintf(fid, '%s', str);
        fclose(fid);
    end
end




function [chi2, chi2List, X, Px, dimX, dimZ] = ...
    linear2dof(numberOfTimeSteps, omegaRScale, omegaQScale, ...
    testProposition4, numObs, obsPeriods, numSubgraph, scenario, sigmaR, sigmaQ)

    import g2o.core.*;
    import pose_graph_experiments.*;
    import two_d_tracking_model_answer.*;

    % Check if obsPeriods is provided, if not, set it to a default value
    if (nargin < 6 || isempty(obsPeriods))
        obsPeriods = 1;
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
%     sigmaR = diag([1, 1]); % Two values for sigmaR
%     sigmaQ = diag([1, 1]); % Two values for sigmaQ

    % Work out the state transition equations for the simulator
    F0=[1 dT; 0 1];
%     Q0=[dT^3/3 dT^2/2;dT^2/2 dT] * sigmaQ;
    Q0=[dT^3/3 dT^2/2;dT^2/2 dT];

    F = [F0 zeros(2); zeros(2) F0];

    % If R is not provided, use default
    if nargin < 9
%         R = sigmaR;
        sigmaR = diag([1, 1]); % Two values for sigmaR
    end

    % If Q is not provided, use default
    if nargin < 10
%         Q = [Q0 zeros(2); zeros(2) Q0];
%         Q = [sigmaQ * Q0(1,1), sigmaQ * Q0(1,2);
%              sigmaQ * Q0(2,1), sigmaQ * Q0(2,2)];
        sigmaQ = diag([1, 1]); % Two values for sigmaQ
    end

    R = sigmaR;
    Q = [sigmaQ * Q0(1,1), sigmaQ * Q0(1,2);
         sigmaQ * Q0(2,1), sigmaQ * Q0(2,2)];

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
                Q1=[dTp^3/3 dTp^2/2;dTp^2/2 dTp];
    
                Qq = [sigmaQ * Q1(1,1), sigmaQ * Q1(1,2);
                     sigmaQ * Q1(2,1), sigmaQ * Q1(2,2)];

                processModelEdge.setF(blkdiag(F1, F1));
                processModelEdge.setInformation(omegaQScale * inv(Qq));
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

