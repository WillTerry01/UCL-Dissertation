close all;
clear all;
clc;

% Number of steps per episode
numberOfTimeSteps = 20;

% Number of episodes
numberOfEpisodes = 2000;

% If set to false, we test proposition 3, which initialises the graph at the
% ground truth value, and does not optimise. If set to true, we test
% proposition 4, which is the distribution after optimising with noisy
% measurements
testProposition4 = true;

% Define the search space for Omega values
variables = [optimizableVariable('omegaRScale', [0.1, 1.9]);
             optimizableVariable('omegaQScale', [0.1, 1.9])];

acquisitionFuncs = {'expected-improvement-per-second-plus', ...
    'expected-improvement', 'expected-improvement-plus', ...
    'expected-improvement-per-second', 'lower-confidence-bound', ...
    'probability-of-improvement'};

acquisitionFunc = acquisitionFuncs{3};
maxObjectiveEvaluations = 100;

% Perform Bayesian optimisation
results = bayesopt(@(x) targetFunction(x, numberOfTimeSteps, numberOfEpisodes, testProposition4), variables, ...
    'AcquisitionFunctionName', acquisitionFunc, 'MaxObjectiveEvaluations', maxObjectiveEvaluations);

% Output the optimal Omega values
disp(results.XAtMinObjective);

% Define the target function, which takes omegaRScale and omegaQScale values, and returns the calculated C value.
% Assume runGPSExample and other necessary functions have been defined and return the required values for computing C.
function cVal = targetFunction(x, numberOfTimeSteps, numberOfEpisodes, testProposition4)
    import g2o.core.*;
    import pose_graph_experiments.*;
    import odometry_model_answer.*;
    
    % Extract the variables
    omegaRScale = x.omegaRScale;
    omegaQScale = x.omegaQScale;

    % Add the code for computing the C value here. For example:
    chi2SumStore = zeros(numberOfEpisodes, 1);
    [chi2SumStore(1), ~, ~, dimX, dimZ] = runGPSExample(numberOfTimeSteps, omegaRScale, omegaQScale, testProposition4);
    parfor r = 2 : numberOfEpisodes
        [chi2SumStore(r), ~] = runGPSExample(numberOfTimeSteps, omegaRScale, omegaQScale, testProposition4);
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