function chi2Plotting(saveResults, testProposition4, os, weekNum, system_name, C_store, meanChi2_store, covChi2_store, numberOfTimeSteps, omegaRScaleArray, omegaQScaleArray, numObs, obsPeriod, numSubgraph)
    
    if nargin < 10
        omegaRScaleArray = 0.1:0.1:1.9; % default value
    end
    if nargin < 11
        omegaQScaleArray = 0.1:0.1:1.9; % default value
    end
    if nargin < 12 || isempty(numObs)
        numObs = numberOfTimeSteps; % default value
    end
    if nargin < 13 || isempty(obsPeriod)
        obsPeriod = 1; % default value
    end
    if nargin < 14
        numSubgraph = 1; % default value
    end

    % Path to store plots
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

    % Build the file names
    C_name = "C_" + system_name;
    Mean_name = "meanChi2_" + system_name;
    Cov_name = "covChi2_" + system_name;

    if size(obsPeriod, 2) ~= 1 || size(obsPeriod, 2) == 1 && obsPeriod ~= 1
        C_name = C_name + "_measurement_rate_" + numObs + "-" + num2str(obsPeriod);
        Mean_name = Mean_name + "_measurement_rate_" + numObs + "-" + num2str(obsPeriod);
        Cov_name = Cov_name + "_measurement_rate_" + numObs + "-" + num2str(obsPeriod);
    end

    if testProposition4
        C_name = C_name + "_prop4";
        Mean_name = Mean_name + "_prop4";
        Cov_name = Cov_name + "_prop4";
    end

    if numSubgraph == 1
        % If there's only one subgraph, we do not add "_subgraph_" to the file and plot names
        figure(1)
        imagesc(omegaRScaleArray, omegaQScaleArray, C_store);
        colorbar;
        xlabel('omegaRScale');
        ylabel('omegaQScale');
        title('Consistency Measurement (C) for different Omega values');
        axis xy;
        if saveResults
            saveas(gcf, basePath + C_name + ".png")
            writematrix(C_store, basePath + C_name + '.csv')
        end

        figure(2)
        imagesc(omegaRScaleArray, omegaQScaleArray, meanChi2_store);
        colorbar;
        xlabel('omegaRScale');
        ylabel('omegaQScale');
        title('Mean Chi2 for different Omega values');
        axis xy;
        if saveResults
            saveas(gcf, basePath + Mean_name + ".png")
            writematrix(meanChi2_store, basePath + Mean_name + '.csv')
        end

        figure(3)
        imagesc(omegaRScaleArray, omegaQScaleArray, covChi2_store);
        colorbar;
        xlabel('omegaRScale');
        ylabel('omegaQScale');
        title('Covariance Chi2 for different Omega values');
        axis xy;
        if saveResults
            saveas(gcf, basePath + Cov_name + ".png")
            writematrix(covChi2_store, basePath + Cov_name + '.csv')
        end
    elseif numSubgraph > 1
        for i = 1:numSubgraph
            C_name_i = C_name + "_subgraph_" + i;
            Mean_name_i = Mean_name + "_subgraph_" + i;
            Cov_name_i = Cov_name + "_subgraph_" + i;

            % Plotting for subgraph i
            figure(i)
            imagesc(omegaRScaleArray, omegaQScaleArray, C_store(:,:,i));
            colorbar; % adds a colorbar, which indicates the scale of C values
            xlabel('omegaRScale');
            ylabel('omegaQScale');
            title(string("Consistency Measurement (C) for Subgraph " + i));
            axis xy; % flips the axis so omegaRScale is x and omegaQScale is y
            if saveResults
                saveas(gcf, basePath + C_name_i + ".png")
                writematrix(C_store(:,:,i), basePath + C_name_i + '.csv')
            end

            % Plot the meanChi2 values for subgraph i
            figure(numSubgraph + i)
            imagesc(omegaRScaleArray, omegaQScaleArray, meanChi2_store(:,:,i));
            colorbar; % adds a colorbar, which indicates the scale of meanChi2 values
            xlabel('omegaRScale');
            ylabel('omegaQScale');
            title(string("Mean Chi2 for Subgraph " + i));
            axis xy; % flips the axis so omegaRScale is x and omegaQScale is y
            if saveResults
                saveas(gcf, basePath + Mean_name_i + ".png")
                writematrix(meanChi2_store(:,:,i), basePath + Mean_name_i + '.csv')
            end

            % Plot the covChi2 values for subgraph i
            figure(2*numSubgraph + i)
            imagesc(omegaRScaleArray, omegaQScaleArray, covChi2_store(:,:,i));
            colorbar; % adds a colorbar, which indicates the scale of covChi2 values
            xlabel('omegaRScale');
            ylabel('omegaQScale');
            title(string("Covariance Chi2 for Subgraph " + i));
            axis xy; % flips the axis so omegaRScale is x and omegaQScale is y
            if saveResults
                saveas(gcf, basePath + Cov_name_i + ".png")
                writematrix(covChi2_store(:,:,i), basePath + Cov_name_i + '.csv')
            end
        end

        % Store the summed C value for each subgraph
        if saveResults
            writematrix(sum(C_store, 3), basePath + C_name + "_sumC.csv");
        end
    end
end
