
clc;
close all;
clear all;

import g2o.core.*;
import one_d_static_model_answer.*;

% Some parameters
numberOfMeasurements = 100;
numberOfRunningTimes = 10;
sigmaR(1) = 1;
sigmaR(2) = 10;

% The information (inverse covariance) for the measurement edge
omegaR = 1 ./ sigmaR;

% Ground truth location
trueX = 10;

% Odd even flag. This causes us to "switch" between the two sensors. The
% first line means we always use the good sensor, the second case
% alternates between the two
oddEven = ones(1, numberOfMeasurements);
% oddEven = mod(0:numberOfMeasurements-1, 2) + 1;

% Store for the average chi2 of the each run
avgChi2Store = NaN(1, numberOfRunningTimes);

for i = 1 : numberOfRunningTimes
    % Sample the noises for the different observations
    z = trueX + sqrt(sigmaR(oddEven)) .* randn(numberOfMeasurements, 1)';

    % Create the graph
    graph = SparseOptimizer();
    algorithm = GaussNewtonOptimizationAlgorithm();
    graph.setAlgorithm(algorithm);
    
    % Create the vertex. This contains the state we want to estimate.
    v = ObjectStateVertex();
    
    % Set the initial estimate. We have to set some initial value. This ideally
    % shouldn't be too far from the final soluton. Here we use the first
    % measurement.
    v.setEstimate(z(1));
    
    % Added the vertex to the graph. The graph now knows that we have some
    % states we want to estimate.
    graph.addVertex(v);
    
    % Store for the covariance and the estimate
%     xStore = NaN(1, numberOfMeasurements);
%     PStore = NaN(1, numberOfMeasurements);
    chi2Store = NaN(1, numberOfMeasurements);

    % Create the edges. Each edge corresponds to a position measurement.
    
    for k = 1 : numberOfMeasurements
    
        % Create the measurement edge
        e = ObjectMeasurementEdge();
    
        % Link it so that it connects to the vertex we want to estimate
        e.setVertex(1, v);
    
        % Set the measurement value and the measurement covariance
        e.setMeasurement(z(k));
        e.setInformation(omegaR(oddEven(k)));
    
        % Add the edge to the graph; the graph now knows we have these edges
        % which need to be added
        graph.addEdge(e);
    
        % Graph construction complete for this iteration
    
        % Initialise the optimization. If you ever change the structure of the
        % graph (e.g., add remove edges and vertices) you must call this before
        % calling optimize. If you don't change the graph structure, you can call
        % optimze multiple times without calling initializeOptimization().
        graph.initializeOptimization();
    
        % Run the optimizer. By default it does at most 10 iterations, but you can
        % pass the number in optionally.
        graph.optimize();
    
        chi2Store(k) = graph.chi2();
    end

    % Store the mean values for each optimized graph
    avgChi2Store(i) = mean(chi2Store);
end

figure(1)
plot(avgChi2Store)
