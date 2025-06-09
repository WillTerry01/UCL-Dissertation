close all;
clear all;
clc;

% Parameters for plotting
os = "win";
weekNum = 13;
system_name = "gps";
saveResults = false;

% Number of steps per episode
numberOfTimeSteps = 20;

% Number of episodes
numberOfEpisodes = 2000;
    
% If set to false, we test proposition 3, which initialises the graph at the
% ground truth value, and does not optimise. If set to true, we test
% proposition 4, which is the distribution after optimising with noisy
% measurements
testProposition4 = true;

% Define the search space for R and Q values
variables = [optimizableVariable('R11', [0.1, 1.9]);
             optimizableVariable('R22', [0.1, 1.9]);
             optimizableVariable('Q11', [0.01, 0.5]);
             optimizableVariable('Q22', [0.01, 0.5]);
             optimizableVariable('Q33', [0.001, 0.05])];

% variables = [optimizableVariable('R11', [0.1, 1.9]);
%              optimizableVariable('R22', [0.1, 1.9])
%             ];

% variables = [optimizableVariable('Q11', [0.01, 0.5]);
%              optimizableVariable('Q22', [0.01, 0.5]);
%              optimizableVariable('Q33', [0.001, 0.05])];

acquisitionFuncs = {'expected-improvement-per-second-plus', ...
    'expected-improvement', 'expected-improvement-plus', ...
    'expected-improvement-per-second', 'lower-confidence-bound', ...
    'probability-of-improvement'};

acquisitionFunc = acquisitionFuncs{1};
maxObjectiveEvaluations = 100;

% Perform Bayesian optimisation
results = bayesopt(@(x) targetFunction(x, numberOfTimeSteps, numberOfEpisodes, testProposition4), variables, ...
    'AcquisitionFunctionName', acquisitionFunc, 'MaxObjectiveEvaluations', maxObjectiveEvaluations);

% Path to store results
if os == "mac"
    basePath = "~/Desktop/week" + weekNum + "/";
else
    basePath = "D:\University\UCL\project\week" + weekNum + "\";
end

% Create folder if not exist and display log
if ~exist(basePath, 'dir')
    mkdir(basePath);
    disp('Folder created successfully.');
else
    disp('Folder already exists.');
end

if testProposition4 == true
    prop = "_prop4";
else
    prop = "";
end

fileName = system_name + "_results_" + ...
    num2str(numberOfTimeSteps) + "-" + num2str(numberOfEpisodes) + ...
    "_" + acquisitionFunc + ...
    "_eval_" + maxObjectiveEvaluations + ...
    prop + ".mat";
if saveResults == true
    save(basePath + fileName, 'results');
end

% Define the target function, which takes R and Q values, and returns the calculated C value.
% Assume runGPSExample and other necessary functions have been defined and return the required values for computing C.
function cVal = targetFunction(x, numberOfTimeSteps, numberOfEpisodes, testProposition4)
    import g2o.core.*;
    import pose_graph_experiments.*;
    import odometry_model_answer.*;
    
    % Extract the variables
    R11 = x.R11;
    R22 = x.R22;
    Q11 = x.Q11;
    Q22 = x.Q22;
    Q33 = x.Q33;

    % Construct R and Q matrices
%     R = eye(2);
    R = diag([R11, R22]);
    Q = diag([Q11, Q22, Q33].^2) ;
%     Q = diag([0.1 0.05 pi/180].^2);

    % Add the code for computing the C value here. For example:
    chi2SumStore = zeros(numberOfEpisodes, 1);
    [chi2SumStore(1), ~, ~, dimX, dimZ] = runGPSExample(numberOfTimeSteps, 1, 1, testProposition4, R, Q);
    parfor r = 2 : numberOfEpisodes
        [chi2SumStore(r), ~] = runGPSExample(numberOfTimeSteps, 1, 1, testProposition4, R, Q);
    end
    meanChi2 = mean(chi2SumStore);
    covChi2 = cov(chi2SumStore);
    if (testProposition4 == true)
        N = dimZ - dimX;
    else
        N = dimZ;
    end
    cVal = abs(log(meanChi2/N)) + abs(log(covChi2/(2*N)));
end

