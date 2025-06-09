clc;
close all;
clear all;

import g2o.core.*;
import two_d_tracking_model_answer.*;

% Some parameters
numberOfTimeSteps = 400;
numberOfRunningTimes = 10;
dT = 1;
sigmaR = 1;
sigmaQ = 100;

% The observation period - observations are available once every
% obsPeriod steps
obsPeriod = 10;

% Work out the state transition equations
F0=[1 dT; 0 1];
Q0=[dT^3/3 dT^2/2;dT^2/2 dT] * sigmaQ;

F = [F0 zeros(2); zeros(2) F0];
Q = [Q0 zeros(2); zeros(2) Q0];
R = eye(2) * sigmaR;

H = [1 0 0 0;
    0 0 1 0];

% Work out the information matrices
omegaR = inv(R);
omegaQ = inv(Q);

% Ground truth array
trueX = zeros(4, numberOfTimeSteps);
z = zeros(2, numberOfTimeSteps);

% Store mean value of chi2
avgChi2Store = NaN(1, numberOfRunningTimes);

for i = 1 : numberOfRunningTimes
    % First timestep
    trueX(2, 1) = 0.1;
    trueX(4, 1) = -0.1;
    z(:, 1) = H * trueX(:, 1) + sqrtm(sigmaR) * randn(2, 1);
    
    % Now predict the subsequent steps
    for k = 2 : numberOfTimeSteps
        trueX(:, k) = F * trueX(:, k - 1) + sqrtm(Q) * randn(4, 1);
        z(:, k) = H * trueX(:, k) + sqrtm(sigmaR) * randn(2, 1);
    end
    
    % Create the graph
    graph = SparseOptimizer();
    algorithm = GaussNewtonOptimizationAlgorithm();
    graph.setAlgorithm(algorithm);
    
    % This array contains the set of vertices for the target state over time
    v = cell(numberOfTimeSteps, 1);
    
    % Store for the covariance and the estimate
    xStore = NaN(1, numberOfTimeSteps);
    PStore = NaN(1, numberOfTimeSteps);
    
    % Now create the vertices and edges
    for n = 1 : numberOfTimeSteps
        % Create the object state vertex
        v{n} = ObjectStateVertex();
    
        % Set the initial estimate to zeros
        v{n}.setEstimate(zeros(4, 1));
        
        % Add the vertex to the graph
        graph.addVertex(v{n});
        
        % If this isn't the first vertex, add the dynamics
        if (n > 1)
            processModelEdge = ObjectProcessModelEdge();
            processModelEdge.setVertex(1, v{n-1});
            processModelEdge.setVertex(2, v{n});
            processModelEdge.setMeasurement([0; 0; 0; 0]);
            processModelEdge.setF(F);
            processModelEdge.setInformation(omegaQ);
            graph.addEdge(processModelEdge);
        end
        
        if (rem(n, obsPeriod) == 1)
            % Create the measurement edge
            e = ObjectMeasurementEdge();
            e.setVertex(1, v{n});
            e.setMeasurement(z(:,n));
            e.setInformation(omegaR);
            graph.addEdge(e);
        end
    end
    
    
    % Initialise the optimization. If you ever change the structure of the
    % graph (e.g., add remove edges and vertices) you must call this before
    % calling optimize. If you don't change the graph structure, you can call
    % optimze multiple times without calling initializeOptimization().
    graph.initializeOptimization();
    
    % Run the optimizer. By default it does at most 10 iterations, but you can
    % pass the number in optionally.
    graph.optimize(5000);
    
    % Get the chi2 value list
    [chi2sum, chi2list] = graph.chi2();
    
    cumulativeSum_chi2List = cumsum(chi2list);
    avgChi2Store(i) = mean(cumulativeSum_chi2List);
end

figure(1)
plot(avgChi2Store)
