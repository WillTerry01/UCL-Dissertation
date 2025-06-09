import g2o.core.*;
import pose_graph_experiments.*;
import two_d_tracking_model_answer.*;

% Number of steps per episode
numberOfTimeSteps = 200;

% Number of episodes
numberOfEpisodes = 1000;

scenario = 1;
    
% If set to false, we test proposition 3, which initialises the graph at the
% ground truth value, and does not optimise. If set to true, we test
% proposition 4, which is the distribution after optimising with noisy
% measurements
testProposition4 = false;

% Parameters to change the frequency of measurement updates
numObs = 50;
obsPeriod = [1 5 10];
% obsPeriod = 1;

% Number of subgraphs
numSubgraph = length(obsPeriod);

% Define the search space for R and Q scales
variables = [optimizableVariable('R11', [0.1, 2]);
             optimizableVariable('R22', [0.1, 2]);
             optimizableVariable('Q11', [1, 3]);
             optimizableVariable('Q22', [1, 3]);];

acquisitionFuncs = {'expected-improvement-per-second-plus', ...
    'expected-improvement', 'expected-improvement-plus', ...
    'expected-improvement-per-second', 'lower-confidence-bound', ...
    'probability-of-improvement'};

acquisitionFunc = acquisitionFuncs{1};
maxObjectiveEvaluations = 100;

% Perform Bayesian optimisation
results = bayesopt(@(x) targetFunction(x, numberOfTimeSteps, numberOfEpisodes, testProposition4, numObs, obsPeriod, numSubgraph, scenario), variables, ...
    'AcquisitionFunctionName', acquisitionFunc, 'MaxObjectiveEvaluations', maxObjectiveEvaluations);

if testProposition4 == true
    prop = "_prop4";
else
    prop = "";
end

% Define the target function, which takes omega scales, and returns the calculated C value.
function cVal = targetFunction(x, numberOfTimeSteps, numberOfEpisodes, testProposition4, numObs, obsPeriod, numSubgraph, scenario)
    import g2o.core.*;
    import pose_graph_experiments.*;
    import two_d_tracking_model_answer.*;
    
    omegaRScale = 1;
    omegaQScale = 1;

    % Extract the variables
    R11 = x.R11;
    R22 = x.R22;
    Q11 = x.Q11;
    Q22 = x.Q22;

    sigmaR = diag([R11, R22]);
    sigmaQ = diag([Q11, Q22]);

    if numSubgraph == 1
        % Compute the number of edges
        numberOfEdges = 2 * numObs - 1 + ...
        floor((numberOfTimeSteps - numObs) / obsPeriod) + ...
        numberOfTimeSteps - numObs;

        chi2Store = zeros(numberOfEpisodes, numberOfEdges);
        chi2SumStore = zeros(numberOfEpisodes, 1);
    else
        chi2Store = zeros(numberOfEpisodes, numSubgraph);
        chi2SumStore = zeros(numSubgraph, 1);
    end

    % First run retrieves the graph dimensions
    [chi2SumStore(1), chi2Store(1, :), ~, ~, dimX, dimZ] = ...
        linear2Dof(numberOfTimeSteps, ...
        omegaRScale, omegaQScale, testProposition4, ...
        numObs, obsPeriod, numSubgraph, scenario, ...
        sigmaR, sigmaQ);
    
    parfor r = 2 : numberOfEpisodes
        [chi2SumStore(r), chi2Store(r, :)] = ...
            linear2Dof(numberOfTimeSteps, ...
            omegaRScale, omegaQScale, testProposition4, ...
            numObs, obsPeriod, numSubgraph, scenario, ...
            sigmaR, sigmaQ);
    end


    if (testProposition4 == true)
        N = dimZ - dimX;
    else
        N = dimZ;
    end

    if numSubgraph == 1
        meanChi2 = mean(chi2SumStore);
        covChi2 = cov(chi2SumStore);
        cVal = abs(log(meanChi2/N)) + abs(log(covChi2/(2*N)));
    else
        % Compute the mean and covariance for each subgraph
        C = zeros(1, numSubgraph);
        for i = 1:numSubgraph
            meanChi2 = mean(chi2Store(:, i));
            covChi2 = cov(chi2Store(:, i));

            % Compute the number of degrees of freedom
            if (testProposition4 == true)
                N = dimZ(i) - dimX(i);
            else
                N = dimZ(i);
            end

            % Compute the Consistency Measurement for each subgraph
            C(i) = abs(log(meanChi2/N)) + abs(log(covChi2/(2*N)));
        end
        cVal = sum(C);
    end
end
