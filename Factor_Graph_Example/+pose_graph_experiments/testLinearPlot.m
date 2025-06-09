close all;
clear all;
clc;

import g2o.core.*;
import pose_graph_experiments.*;
import two_d_tracking_model_answer.*;

% Parameters for plotting
os = "win";
weekNum = 16;
system_name = "linear";
saveResults = true;
save2txt = true;

numberOfTimeSteps = 200; % Number of steps per episode
numberOfEpisodes = 1000; % Number of episodes
scenario = 1;

basePath = "D:\University\UCL\project\week" + weekNum + "\";
fileName = "C_GT_scenario_" + scenario + ".txt";
filePath = basePath + fileName;
if exist(filePath, 'file')
    % Open the file in write mode. This will clear its content.
    fid = fopen(filePath, 'w');
    % Close the file
    fclose(fid);
end

% If set to false, we test proposition 3, which initializes the graph at the
% ground truth value, and does not optimize. If set to true, we test
% proposition 4, which is the distribution after optimizing with noisy
% measurements
testProposition4 = false;

% Parameters to change the frequency of measurement updates
numObs = 50;
obsPeriod = [1 5 10];

% Number of subgraphs
numSubgraph = length(obsPeriod);

% Define the range and step for Omega scales
omegaRScaleArray = 0.1:0.02:1.9;
omegaQScaleArray = 0.1:0.02:1.9;

[RM, QM] = meshgrid(omegaRScaleArray, omegaQScaleArray);

% Create a matrix to store C values
C_store = zeros(length(omegaRScaleArray), length(omegaQScaleArray), numSubgraph);

% Create matrices to store meanChi2 and covChi2 values
meanChi2_store = zeros(length(omegaRScaleArray), length(omegaQScaleArray), numSubgraph);
covChi2_store = zeros(length(omegaRScaleArray), length(omegaQScaleArray), numSubgraph);

% Create temporary variables to store results for each iteration
tempC_store = cell(numel(RM), 1);
tempMeanChi2_store = cell(numel(RM), 1);
tempCovChi2_store = cell(numel(RM), 1);

% Loop over all possible Omega values
parfor i = 1:numel(RM)
        
    % Initialize chi2Store for each episode
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

    % Get chi2Sum and chi2 values, along with the edges in graph for each
    % running episode
    fprintf('OmegaRScale: %.2f, OmegaQScale: %.2f\n', ...
        RM(i), QM(i))

    % First run retrieves the graph dimensions
    [chi2SumStore(1), chi2Store(1, :), ~, ~, dimX, dimZ] = ...
        runLinearExample(numberOfTimeSteps, ...
        RM(i), QM(i), testProposition4, ...
        numObs, obsPeriod, numSubgraph, scenario);

    for r = 2 : numberOfEpisodes
        [chi2SumStore(r), chi2Store(r, :)] = ...
            runLinearExample(numberOfTimeSteps, ...
            RM(i), QM(i), testProposition4, ...
            numObs, obsPeriod, numSubgraph, scenario);
    end
    
    % Convert linear index to matrix indices
    [idxR, idxQ] = ind2sub(size(RM), i);

    % Store C, meanChi2, and covChi2 in matrices for each subgraph
    tempC_store{i} = zeros(1, 1, numSubgraph);
    tempMeanChi2_store{i} = zeros(1, 1, numSubgraph);
    tempCovChi2_store{i} = zeros(1, 1, numSubgraph);

    for j = 1:numSubgraph
        % Calculate meanChi2, covChi2
        meanChi2 = mean(chi2Store(:, j));
        covChi2 = cov(chi2Store(:, j));

        % Compute the number of degrees of freedom
        if (testProposition4 == true)
            N = dimZ(j) - dimX(j);
        else
            N = dimZ(j);
        end

        % Store C, meanChi2, and covChi2 in matrices
        tempC_store{i}(1, 1, j) = abs(log(meanChi2/N)) + abs(log(covChi2/(2*N)));
        tempMeanChi2_store{i}(1, 1, j) = meanChi2;
        tempCovChi2_store{i}(1, 1, j) = covChi2;
    end
end

% Update 3D matrices after the parfor loop
for i = 1:numel(RM)
    [idxR, idxQ] = ind2sub(size(RM), i);
    C_store(idxR, idxQ, :) = tempC_store{i};
    meanChi2_store(idxR, idxQ, :) = tempMeanChi2_store{i};
    covChi2_store(idxR, idxQ, :) = tempCovChi2_store{i};

    if save2txt
        % Slightly cheesy way to append to an existing file
        fid = fopen(filePath, 'a+');
        str = strjoin({sprintf('%.2f %.2f', QM(i), RM(i)), sprintf(' %d', C_store(idxR, idxQ, :)), sprintf(' %d\n', sum(C_store(idxR, idxQ, :)))});
        fprintf(fid, '%s', str);
        fclose(fid);
    end
end


% Display results
% Plotting
chi2Plotting(saveResults, testProposition4, os, weekNum, system_name,...
    C_store, meanChi2_store, covChi2_store, ...
    numberOfTimeSteps, omegaRScaleArray, omegaQScaleArray, ...
    numObs, obsPeriod, numSubgraph);

