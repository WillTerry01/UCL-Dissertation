close all;
clear all;
clc;

import g2o.core.*;
import pose_graph_experiments.*;
import two_d_tracking_model_answer.*;

% Number of steps per episode
numberOfTimeSteps = 200;

% Number of episodes
numberOfEpisodes = 1000;

scenario = 1;

% RM = 0.1 : 0.05 : 1.9;
% QM = 0.1 : 0.05 : 1.9;


RM = 1;
QM = 1;

% basePath = "D:\University\UCL\project\";
% fileName = "frob_results.txt";
% filePath = basePath + fileName;
% if exist(filePath, 'file')
%     % Open the file in write mode. This will clear its content.
%     fid = fopen(filePath, 'w');
%     % Close the file
%     fclose(fid);
% end

% Omega Scales
for omegaRScale = RM
    for omegaQScale = QM
    
    % Parameters to change the frequency of measurement updates
    numObs = 50;
    obsPeriod = 1;
    
    % Number of subgraphs
    numSubgraph = length(obsPeriod);
    
    % If set to false, we test proposition 3, which initializes the graph at the
    % ground truth value, and does not optimize. If set to true, we test
    % proposition 4, which is the distribution after optimizing with noisy
    % measurements
    testProposition4 = false;
    
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
    
    % First run retrieves the graph dimensions
    [chi2SumStore(1), chi2Store(1, :), X, Px, dimX, dimZ] = ...
        runLinearExample(numberOfTimeSteps, ...
        omegaRScale, omegaQScale, testProposition4, ...
        numObs, obsPeriod, numSubgraph, scenario);
    
    parfor r = 2 : numberOfEpisodes
        fprintf('%03d\n', r)
        [chi2SumStore(r), chi2Store(r, :)] = ...
            runLinearExample(numberOfTimeSteps, ...
            omegaRScale, omegaQScale, testProposition4, ...
            numObs, obsPeriod, numSubgraph, scenario);
    end
    
    if numSubgraph == 1
        % Compute the number of degrees of freedom
        if (testProposition4 == true)
            N = dimZ - dimX;
        else
            N = dimZ;
        end
    
        % This figure works out and plots the mean and covarianace for the sum of
        % the chi2 values over all edges. Propositions 3 and 4 are guaranteed to
        % apply to ONLY these values.
        figure(1)
        plot(chi2SumStore)
        meanChi2 = mean(chi2SumStore);
        covChi2 = cov(chi2SumStore);
        title(sprintf('Mean: %f; Covariance %f', meanChi2, covChi2))
        
        Px = full(Px{1});
        X = X{1};
  
        % Compute the Consistency Measurement
        C = abs(log(meanChi2/N)) + abs(log(covChi2/(2*N)));
        J = abs(log(meanChi2/N));
    else
        % Compute the mean and covariance for each subgraph
        C = zeros(1, numSubgraph);
        for i = 1:numSubgraph
            figure(i)
            plot(chi2Store(:, i))
            meanChi2 = mean(chi2Store(:, i));
            covChi2 = cov(chi2Store(:, i));
            title(sprintf('Subgraph: %d, Mean: %f; Covariance %f', i, meanChi2, covChi2))
        
            % Compute the number of degrees of freedom
            if (testProposition4 == true)
                N = dimZ(i) - dimX(i);
            else
                N = dimZ(i);
            end
    
            % Compute the Consistency Measurement for each subgraph
            C(i) = abs(log(meanChi2/N)) + abs(log(covChi2/(2*N)));
        end
    end
% %     Slightly cheesy way to append to an existing file
%     fid = fopen('D:\University\UCL\project\results_test.txt', 'a+');
%     str = strjoin({sprintf('%d %d', omegaQScale, omegaRScale), sprintf(' %d', C), sprintf(' %d\n', sum(C))});
%     fprintf(fid, '%s', str);
%     fclose(fid);
    end
end

