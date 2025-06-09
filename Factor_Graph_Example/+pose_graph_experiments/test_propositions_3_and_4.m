% Test of propositions 3 and 4 on a linear example which consists of a
% target.
close all;
clear all;
clc;

import g2o.core.*;
import pose_graph_experiments.*;
import two_d_tracking_model_answer.*;

% Number of steps per episode
numberOfTimeSteps = 100;

% Nuumber of episodes
numberOfEpisodes = 2000;

% If set to false, we test proposition 3, which initializes the graph at the
% ground truth value, and does not optimize. If set to true, we test
% proposition 4, which is the distribution after optimizing with noisy
% measurements
testProposition4 = true;

chi2Store = zeros(numberOfEpisodes, 2 * numberOfTimeSteps - 1);

% This uses the matlab parallel toolbox which can speed up running the code
% significantly
parfor r = 1 : numberOfEpisodes
    fprintf('%03d\n', r)
    chi2Store(r, :) = runExample(numberOfTimeSteps, 1, 1, testProposition4);
end

% Now plot the results. The first two figures show the edge-by-edge chi2
% values. Note that propositions 3 and 4 actually say nothing about
% edge-by-edge behaviour. They only describe the behaviour of the sum of
% all of the chi2 values over all of the edges. (For proposition 3, the
% errors are all independent on an edge-by-edge basis, and so the graphs
% are still useful.)

figure(1)
% TimestepMean calculates the average chi2 value for a specific edge in the
% graph for all run. Not the average chi2 for all edges in the graph in a
% single run. 
timestepMean = mean(chi2Store,1);
hold off
plot(timestepMean)
hold on
plot(movmean(timestepMean, numberOfTimeSteps), 'LineWidth', 4)
title('Edge-by-Edge mean chi2')

figure(2)
timestepCov = mean(chi2Store.^2,1) - timestepMean.^2;
hold off
plot(timestepCov)
hold on

plot(movmean(timestepCov, numberOfTimeSteps), 'LineWidth', 4)
title('Edge-by-Edge covariance chi2')

% This figure works out and plots the mean and covarianace for the sum of
% the chi2 values over all edges. Propositions 3 and 4 are guaranteed to
% apply to ONLY these values.

figure(3)
chi2Sum = sum(chi2Store,2);

figure(3)
hold off
plot(chi2Sum)
hold on

title(sprintf('Mean: %f; Covariance %f', mean(chi2Sum), cov(chi2Sum)))

%keyboard


function chi2List = runExample(numberOfTimeSteps, omegaRScale, omegaQScale, testProposition4)

    import g2o.core.*;
    import two_d_tracking_model_answer.*;
    
    fprintf('runExample\n')
    
    % Some parameters
    
    dT = 1;
    sigmaR = 10;
    sigmaQ = 1;
    
    % Work out the state transition equations
    F0=[1 dT; 0 1];
    Q0=[dT^3/3 dT^2/2;dT^2/2 dT] * sigmaQ;
    
    F = [F0 zeros(2); zeros(2) F0];
    Q = [Q0 zeros(2); zeros(2) Q0];
    R = eye(2) * sigmaR;
    
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
    
    % Create the graph
    graph = SparseOptimizer();
    algorithm = GaussNewtonOptimizationAlgorithm();
    graph.setAlgorithm(algorithm);
    
    % This array contains the set of vertices for the target state over time
    v = cell(numberOfTimeSteps, 1);
    
    % Now create the vertices and edges
    
    for n = 1 : numberOfTimeSteps
        
        % Create the first object state vertex
        v{n} = ObjectStateVertex();
    
        % Set the initial estimate.
        if (testProposition4 == false)
            v{n}.setEstimate(trueX(:, n));
        else
            v{n}.setEstimate(0*trueX(:, n));
        end
        
        % Added the vertex to the graph.
        graph.addVertex(v{n});
        
        % If this isn't the first vertex, add the dynamics
        if (n > 1)
            processModelEdge = ObjectProcessModelEdge();
            processModelEdge.setVertex(1, v{n-1});
            processModelEdge.setVertex(2, v{n});
            processModelEdge.setMeasurement([0;0;0;0]);
            processModelEdge.setF(F);
            processModelEdge.setInformation(omegaQ);
            graph.addEdge(processModelEdge);
        end
        
        % Create the measurement edge
        e = ObjectMeasurementEdge();
        
        % Link it so that it connects to the vertex we want to estimate
        e.setVertex(1, v{n});
        
        % Set the measurement value and the measurement covariance
        e.setMeasurement(z(:,n));
        e.setInformation(omegaR);
        
        % Add the edge to the graph; the graph now knows we have these edges
        % which need to be added
        graph.addEdge(e);
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
    [chi2,chi2List] = graph.chi2();

end
