close all;
clear all;
clc;

import g2o.core.*;
import pose_graph_experiments.*;
import odometry_model_answer.*;

% Number of steps per episode
numberOfTimeSteps = 20;

% Number of episodes
numberOfEpisodes = 100;

% Number of landmarks and layout
numberOfLandmarks = 200;
extent = 300;

% Populate the landmark locations
landmarks = (rand([2 numberOfLandmarks]) - 0.5) * extent;

% Omega Scales
omegaRScale = 1;
omegaQScale = 1;

% If set to false, we test proposition 3, which initializes the graph at the
% ground truth value, and does not optimize. If set to true, we test
% proposition 4, which is the distribution after optimizing with noisy
% measurements
testProposition4 = false;

% Store the results into cell
% results = cell(numberOfEpisodes, 1);

% Store chi2 and dims
chi2SumStore = zeros(numberOfEpisodes, 1);
dimXStore = zeros(numberOfEpisodes, 1);
dimZStore = zeros(numberOfEpisodes, 1);

% [chi2, chi2List, edges, dimX, dimZ] = ...
%     runVehicleTrackingExample(numberOfTimeSteps, ...
%     omegaRScale, omegaQScale, testProposition4);
% results{1} = struct('chi2', chi2, 'chi2List', chi2List, 'edges', edges, 'dimX', dimX, 'dimZ', dimZ);

% parfor i = 1 : numberOfEpisodes
%     [chi2, chi2List, ~, dimX, dimZ] = ...
%         runVehicleTrackingExample(numberOfTimeSteps, ...
%         omegaRScale, omegaQScale, testProposition4, landmarks);
% 
%     results{i} = struct('chi2', chi2, 'chi2List', chi2List, 'dimX', dimX, 'dimZ', dimZ);
% 
%     chi2SumStore{i} = chi2;
%     dimXStore{i} = dimX;
%     dimZStore{i} = dimZ;
% end

parfor i = 1 : numberOfEpisodes
    fprintf("%03d", i);
    [chi2SumStore(i), ~, ~, dimXStore(i), dimZStore(i)] = ...
        runVehicleTrackingExample(numberOfTimeSteps, ...
        omegaRScale, omegaQScale, testProposition4, landmarks);
end

% Compute the number of degrees of freedom
% dimX = cellfun(@mean, dimXStore);
dimX = mean(dimXStore);
dimZ = mean(dimZStore);
% dimZ = cellfun(@mean, dimZStore);
% dimZ = dimZStore;
% dimX = dimXStore;

if (testProposition4 == true)
    N = dimZ - dimX;
else
    N = dimZ;
end

% This figure works out and plots the mean and covarianace for the sum of
% the chi2 values over all edges. Propositions 3 and 4 are guaranteed to
% apply to ONLY these values.
figure(1)
% hold on;
% for i = 1:numel(chi2SumStore)
%     plot(chi2SumStore{i});
% end
% hold off;
plot(chi2SumStore);
% meanChi2 = cellfun(@mean, chi2SumStore);
% covChi2 = cellfun(@cov, chi2SumStore, 'UniformOutput', false);
meanChi2 = mean(chi2SumStore);
covChi2 = cov(chi2SumStore);
title(sprintf('Mean: %f; Covariance %f', meanChi2, covChi2))

% Compute the Consistency Measurement
C = abs(log(meanChi2/N)) + abs(log(covChi2/(2*N)));