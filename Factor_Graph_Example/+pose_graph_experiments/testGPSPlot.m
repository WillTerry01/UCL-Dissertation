close all;
clear all;
clc;

import g2o.core.*;
import pose_graph_experiments.*;
import odometry_model_answer.*;

% Parameters for plotting
os = "win";
weekNum = 16;
system_name = "gps";
saveResults = true;
save2txt = true;

% Number of steps per episode
numberOfTimeSteps = 20;

% Number of episodes
numberOfEpisodes = 2000;

basePath = "D:\University\UCL\project\week" + weekNum + "\";
fileName = "C_GT_GPS.txt";
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
testProposition4 = true;

% Parameters to change the frequency of measurement updates
numObs = 100;
obsPeriod = 1;

% Define the range and step for Omega scales
omegaRScaleArray = 0.1:0.02:1.9;
omegaQScaleArray = 0.1:0.02:1.9;

[RM, QM] = meshgrid(omegaRScaleArray, omegaQScaleArray);

% Create a matrix to store C values
C_store = zeros(length(omegaRScaleArray), length(omegaQScaleArray));

% Create matrices to store meanChi2 and covChi2 values
meanChi2_store = zeros(length(omegaRScaleArray), length(omegaQScaleArray));
covChi2_store = zeros(length(omegaRScaleArray), length(omegaQScaleArray));

% Compute the number of edges
numberOfEdges = 2 * numObs - 1 + ...
    floor((numberOfTimeSteps - numObs) / obsPeriod) + ...
    numberOfTimeSteps - numObs;

% Loop over all possible Omega values
parfor i = 1:numel(RM)
        
    % Initialize chi2Store for each episode
    chi2Store = zeros(numberOfEpisodes, numberOfEdges);
    chi2SumStore = zeros(numberOfEpisodes, 1);
%     edgeStore = cell(numberOfEpisodes, 1);

    % Get chi2Sum and chi2 values, along with the edges in graph for each
    % running episode
    fprintf('OmegaRScale: %.2f, OmegaQScale: %.2f\n', ...
        RM(i), QM(i))

    % First run retrieves the graph dimensions
    [chi2SumStore(1), chi2Store(1, :), ~, dimX, dimZ] = ...
        runGPSExample(numberOfTimeSteps, ...
        RM(i), QM(i), testProposition4);

    for r = 2 : numberOfEpisodes
        [chi2SumStore(r), chi2Store(r, :)] = ...
            runGPSExample(numberOfTimeSteps, ...
            RM(i), QM(i), testProposition4);
    end

    % Calculate meanChi2, covChi2
    meanChi2 = mean(chi2SumStore);
    covChi2 = cov(chi2SumStore);

    % Compute the number of degrees of freedom
    if (testProposition4 == true)
        N = dimZ - dimX;
    else
        N = dimZ;
    end

    % Store C, meanChi2, and covChi2 in matrices
    C_store(i) = abs(log(meanChi2/N)) + abs(log(covChi2/(2*N)));
    meanChi2_store(i) = meanChi2;
    covChi2_store(i) = covChi2;
end

% save C values to txt file
if save2txt
    for i = 1:numel(RM)
        [idxR, idxQ] = ind2sub(size(RM), i);
        % Slightly cheesy way to append to an existing file
        fid = fopen(filePath, 'a+');
        str = strjoin({sprintf('%.2f %.2f', QM(i), RM(i)), sprintf(' %d\n', C_store(idxR, idxQ, :))});
        fprintf(fid, '%s', str);
        fclose(fid);
    end
end

%delete(gcp('nocreate')); % stop the parallel pool

% Plotting
chi2Plotting(saveResults, testProposition4, os, weekNum, system_name,...
    C_store, meanChi2_store, covChi2_store, ...
    numberOfTimeSteps, omegaRScaleArray, omegaQScaleArray);