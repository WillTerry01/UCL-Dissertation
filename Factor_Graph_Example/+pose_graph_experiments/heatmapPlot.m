% clear all;
% close all;
% clc;

omegaRScaleArray = 0.1:0.02:1.9;
omegaQScaleArray = 0.1:0.02:1.9;

filePaths = [
    "D:\University\UCL\project\week16\prob3_scenario_1_obs=1\C_linear.csv"
    "D:\University\UCL\project\week16\prop4_scenario_1_obs=1\C_linear_prop4.csv"
    "D:\University\UCL\project\week16\gps_20-2000_prop3\C_gps.csv"
    "D:\University\UCL\project\week16\gps_20-2000_prop4\C_gps_prop4.csv"
    "D:\University\UCL\project\week16\prop3_scenario_1\C_linear_measurement_rate_50-1   5  10_sumC.csv"
    %"D:\University\UCL\project\week16\prop3_scenario_2\C_linear_measurement_rate_50-1   5  10_sumC.csv"
    "D:\University\UCL\project\week15\scenario_1\C_linear_measurement_rate_50-1   5  10_prop4_sumC.csv" 
    %"D:\University\UCL\project\week15\scenario_2\C_linear_prop4_sumC.csv" 
            ];

for i = 1: length(filePaths)
    generateHeatmap(filePaths(i), omegaRScaleArray, omegaQScaleArray);
end


function generateHeatmap(filePath, omegaRScaleArray, omegaQScaleArray)
    % Load data from the given file
    matrixData = readmatrix(filePath);
    
    % Find the minimum value and its location in the matrix
    [minVal, index] = min(matrixData(:));
    [row, col] = ind2sub(size(matrixData), index);
    [~, fileName, ~] = fileparts(filePath);
    titleName = strrep(fileName, '_', ' ');

    % Display heatmap
    figure;
    imagesc(omegaRScaleArray, omegaQScaleArray, matrixData);
    xlabel('omegaRScale');
    ylabel('omegaQScale');
    title(titleName)
    colorbar;
    
    % Define colormap properties
    numColors = 256;
    halfWay = floor(numColors * 1/20);

    % Create the first half of the colormap transitioning from dark blue to yellow
    firstHalf = [linspace(0, 1, halfWay)', linspace(0, 1, halfWay)', ones(halfWay, 1)];

    % Create the second half of the colormap transitioning from yellow to red
    secondHalf = [ones(numColors-halfWay, 1), linspace(1, 0.5, numColors-halfWay)', linspace(1, 0, numColors-halfWay)'];

    customColormap = [firstHalf; secondHalf];
    colormap(customColormap);

%     % Mark the minimum value location with a yellow cross
%     hold on;
%     plot(omegaRScaleArray(col), omegaQScaleArray(row), '+', 'MarkerSize', 10, 'LineWidth', 1.5 , 'Color', 'red'); 
end

