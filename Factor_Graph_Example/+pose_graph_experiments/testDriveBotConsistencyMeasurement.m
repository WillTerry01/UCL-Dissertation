clear all;
close all;
clc;

import pose_graph_experiments.*;

% Parameters
numberOfEpisodes = 10;
omegaRScale = 1; % GPS
omegaQScale = 1; % odemetry

% Create the configuration object.
configuration = drivebot.SimulatorConfiguration();

% Since we are doing prediction and GPS, disable the SLAM sensor
configuration.enableGPS = true;

% Set to true for part ii
configuration.enableCompass = false;

scenarioDirectory = 'q1_e';

chi2SumStore = zeros(numberOfEpisodes, 1);

tic;
% First run retrieves the graph dimensions
[chi2SumStore(1), dimX, dimZ] = ...
    runDriveBotExample(configuration, scenarioDirectory, omegaRScale, omegaQScale);

% Get chi2Sum and chi2 values,
parfor r = 2 : numberOfEpisodes
    fprintf('%03d\n', r)
    chi2SumStore(r) = ...
        runDriveBotExample(configuration, scenarioDirectory, omegaRScale, omegaQScale);
end
toc;

% Get the elapsed time in seconds
elapsedTimeInSeconds = toc;

% Convert elapsed time to HH:MM:SS format
timeString = datestr(seconds(elapsedTimeInSeconds), 'HH:MM:SS');

disp(['Elapsed time: ', timeString]);

N = dimZ - dimX;

figure(2)
plot(chi2SumStore)
meanChi2 = mean(chi2SumStore);
covChi2 = cov(chi2SumStore);
title(sprintf('Mean: %f; Covariance %f', meanChi2, covChi2))

% Compute the Consistency Measurement
C = abs(log(meanChi2/N)) + abs(log(covChi2/(2*N)));