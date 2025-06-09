clear all;
close all;
clc;

% Number of steps per episode
numberOfTimeSteps = 20;
% numberOfEpisodes = 5000;
numberOfEpisodes = 5000;

% Omega Scales
omegaRScale = 1;
omegaQScale = 1;
% omegaRScale = 0.98672;
% omegaQScale = 0.97918;
% omegaRScale = 0.1;
% omegaQScale = 2;
% omegaRScale = 2;
% omegaQScale = 0.1;


% If set to false, we test proposition 3, which initializes the graph at the
% ground truth value, and does not optimize. If set to true, we test
% proposition 4, which is the distribution after optimizing with noisy
% measurements
testProposition4 = true;

% Initialise accumulators for mean error, MSE, and mean covariance
meanErrorAcc = zeros(3, numberOfTimeSteps);
mseAcc = zeros(numberOfTimeSteps, 1);
meanCovarianceAcc = zeros(numberOfTimeSteps, 1);

chi2Store = zeros(numberOfEpisodes, 2 * numberOfTimeSteps - 1);
chi2SumStore = zeros(numberOfEpisodes, 1);

% R = diag([1.2939 1.1973]);
% Q = diag([0.4859 0.49986 0.038036] .^2);

% % GPS measurement covariance
% R = eye(2);
% % Odometry covariance
% Q = diag([0.1 0.05 pi/180].^2);

Px = cell(numberOfEpisodes, 1);
X = cell(numberOfEpisodes, 1);
trueState = cell(numberOfEpisodes, 1);

% First run retrieves the graph dimensions
[~, ~, X(1,:), Px(1,:), ~, ~, trueState{1}] = ...
    gpsPerformance(numberOfTimeSteps, ...
    omegaRScale, omegaQScale, testProposition4);

% Get chi2Sum and chi2 values,
parfor r = 2 : numberOfEpisodes
    fprintf('%03d\n', r)
    [~, ~, X(r,:), Px(r,:), ~, ~, trueState{r}] = ...
        gpsPerformance(numberOfTimeSteps, ...
        omegaRScale, omegaQScale, testProposition4);
end

for i = 1:numberOfEpisodes
    X{i,:} = reshape(vertcat(X{i, :}), 3, numberOfTimeSteps);  % Reshape X
    covMatrix = Px{i};  % Retrieve the covariance matrix for this episode
    
    for t = 1:numberOfTimeSteps
        errorMatrix = X{i}(:, t) - trueState{i}(:, t);  % Assuming trueX is 4xnumberOfTimeSteps
        
        % Extract the corresponding 3x3 sub-matrix from the covariance matrix
        subCov = covMatrix((3*(t-1)+1):(3*t), (3*(t-1)+1):(3*t));
        
        meanErrorAcc(:, t) = meanErrorAcc(:, t) + errorMatrix;
        mseAcc(t) = mseAcc(t) + mean(errorMatrix.^2);
        
        % Average the diagonal elements of the 3x3 sub-matrix to get an approximation of variance
        meanCovarianceAcc(t) = meanCovarianceAcc(t) + mean(diag(subCov));
    end
end

meanError = meanErrorAcc / numberOfEpisodes;
meanMSE = mseAcc / numberOfEpisodes;
meanCovariance = meanCovarianceAcc / numberOfEpisodes;

writematrix(meanError, "D:\University\UCL\project\week17\gps_mean_Error_" + ...
    num2str(omegaRScale) + '_' + num2str(omegaQScale) ...
    + '.csv')

writematrix(meanMSE, "D:\University\UCL\project\week17\gps_mean_MSE_" + ...
    num2str(omegaRScale) + '_' + num2str(omegaQScale) ...
    + '.csv')

writematrix(meanCovariance, "D:\University\UCL\project\week17\gps_mean_Covariance_" + ...
    num2str(omegaRScale) + '_' + num2str(omegaQScale) ...
    + '.csv')

figure;
subplot(2,1,1);
plot(meanMSE);
title('State Estimation Mean Squared Error');
xlabel('Time Step');
ylabel('MSE');

subplot(2,1,2);
plot(meanCovariance);
title('Mean Covariance (Diagonal Average)');
xlabel('Time Step');
ylabel('Covariance');


function [chi2, chi2List, X, Px, dimX, dimZ, trueX] = gpsPerformance(numberOfTimeSteps, omegaRScale, omegaQScale, testProposition4, R, Q)
    import g2o.core.*;
    import g2o.stuff.*;
    import pose_graph_experiments.*;
    import odometry_model_answer.*;
    
    % Nominal odometry
    odometry=[1 0 pi/180]';
    
    if nargin < 5  % If R and Q are not passed as parameters
        % GPS measurement covariance
        R = eye(2);
        % Odometry covariance
        Q = diag([0.1 0.05 pi/180].^2);
    else  % If R and Q are passed as parameters
        omegaRScale = 1;
        omegaQScale = 1;
    end

    % Work out the information matrices
    omegaR = omegaRScale * inv(R);
    omegaQ = omegaQScale * inv(Q);
    
    % Ground truth array
    trueX = zeros(3, numberOfTimeSteps);
    
    % Position observations
    z = zeros(2, numberOfTimeSteps);
    
    % Observation matrix
    H = [1 0 0;0 1 0];
    
    % First timestep
    z(:, 1) = H * trueX(:, 1) + sqrtm(R) * randn(2, 1);
    
    % Now predict the subsequent steps
    for k = 2 : numberOfTimeSteps
        priorX = trueX(:, k-1);
        v = sqrtm(Q) * randn(3, 1);
        M = [cos(priorX(3)) -sin(priorX(3));
            sin(priorX(3)) cos(priorX(3))];
        predictedX = priorX;
        predictedX(1:2) = predictedX(1:2) + M * (odometry(1:2) + v(1:2));
        predictedX(3) = g2o.stuff.normalize_theta(predictedX(3) + odometry(3) + v(3));
        trueX(:, k) = predictedX;
        z(:, k) = H * trueX(:, k) + sqrtm(R) * randn(2, 1);
    end
    
    % Create the graph
    graph = SparseOptimizer();
    algorithm = LevenbergMarquardtOptimizationAlgorithm();
    graph.setAlgorithm(algorithm);
    
    % This array contains the set of vertices for the target state over time
    vertices = cell(numberOfTimeSteps, 1);
    
    % This is the prior we compute for the graph. Unlike the true value above
    % (which uses the real odometry), the prior here is computed using the
    % nominal odometry information
    X0 = zeros(3, numberOfTimeSteps);
    
    % Now create the vertices and edges
    
    for k = 1 : numberOfTimeSteps
        
        % Create the object state vertex
        vertices{k} = VehicleStateVertex();

        % Added the vertex to the graph.
        graph.addVertex(vertices{k});
        
        % If this isn't the first vertex, predict the nominal value and add an
        % edge
        if (k > 1)
            % Create the edge
            processModelEdge = VehicleKinematicsEdge();
            processModelEdge.setVertex(1, vertices{k-1});
            processModelEdge.setVertex(2, vertices{k});
            processModelEdge.setMeasurement(odometry);
            processModelEdge.setInformation(omegaQ);
            graph.addEdge(processModelEdge);
            
            % Predict the state for the initial condition
            priorX = X0(:, k-1);
            M = [cos(priorX(3)) -sin(priorX(3));
                sin(priorX(3)) cos(priorX(3))];
            predictedX = priorX;
            predictedX(1:2) = predictedX(1:2) + M * odometry(1:2);
            predictedX(3) = g2o.stuff.normalize_theta(predictedX(3) + odometry(3));
            X0(:, k) = predictedX;
        end
                
        % Set the initial estimate.
%         vertices{k}.setEstimate(X0(:,k));
        if (testProposition4 == false)
            vertices{k}.setEstimate(trueX(:, k));
        else
            vertices{k}.setEstimate(X0(:,k));
        end
        
        % Create the measurement edge
        e = GPSMeasurementEdge();
        
        % Link it so that it connects to the vertex we want to estimate
        e.setVertex(1, vertices{k});
        
        % Set the measurement value and the measurement covariance
        e.setMeasurement(z(:,k));
        e.setInformation(omegaR);
        
        % Add the edge to the graph
        graph.addEdge(e);
    end
    
    % Graph construction complete
    
    % Initialise the optimization. This is done here because it's a bit
    % expensive and if we cache it, we can run multiple optimizations without
    % having to build everything from scratch
    graph.initializeOptimization();

    % Optimize
    if (testProposition4 == true)
        graph.optimize();
    end

    % Compute the chi2 value
    [chi2,chi2List] = graph.chi2();
    
    X = cell(1, 1);
    Px = cell(1, 1);

    if (nargout == 7)
        dimX = 0;
        dimZ = 0;
        edges = graph.edges();
        vertices = graph.vertices();

        % Get the covariance of vertices
        [X{1}, Px{1}] = graph.computeMarginals();
        
        for e = 1 : length(edges)
            dimZ = dimZ + edges{e}.dimension();
        end

        for v =  1 : length(vertices)
            dimX = dimX + vertices{v}.dimension();
        end
    end
end