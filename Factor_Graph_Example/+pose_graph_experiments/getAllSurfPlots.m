function getAllSurfPlots(os, weekNum, omegaRScaleArray, omegaQScaleArray)

    disp("Plotting ...")
    % Path to read files
    if os == "mac"
        basePath = "~/Desktop/week" + weekNum + "/";
    else
        basePath = "D:\\University\\UCL\\project\\week" + weekNum + "\\";
    end

    % Get all CSV files recursively in the folder and subfolders
    allFiles = getAllCsvFiles(basePath);

    % Initialise processed files list
    processedFiles = {};

    % Loop through all files
    for i = 1:numel(allFiles)
        filePath = allFiles{i};
        [~, fileName, ~] = fileparts(filePath);

        % Only consider files starting with "C_"
        if startsWith(fileName, 'C_')
            
            % Read values
            C = readmatrix(filePath);
            
            % Extract system_name from the file name
            splitName = split(fileName, '_');
%             if numel(splitName) == 2
%                 % If there is only one underscore, extract the substring between underscore and ".asv"
%                 system_name = extractBefore(splitName{2}, '.');
%             else
%                 % If there are two or more underscores, extract the string between the first and second underscore
%                 system_name = splitName{2};
%             end
            system_name = splitName{2};
            % Handle the capitalization for system_name
            if system_name == "gps"
                system_name = string(upper(system_name));
            else
                system_name = string(upper(system_name(1))) + string(lower(system_name(2:end)));
            end

            % Check if file name contains 'prop4'
            if contains(fileName, 'prop4')
                prop = ' (Proposition 4)';
            else
                prop = ' (Proposition 3)';
            end
            
            % Check if file name contains 'measurement_rate'
            if contains(fileName, 'measurement_rate')
                splitRate = split(splitName{5}, '-');
                numObs = splitRate{1};
                obsPeriod = splitRate{2};
                measure_rate = " Measurement Rate " + num2str(numObs) + "/" + num2str(obsPeriod);
            else
                measure_rate = "";
            end

            % Plotting
            figure(length(processedFiles) + 1)
            surf(omegaRScaleArray, omegaQScaleArray, C);
            colorbar; % adds a colorbar, which indicates the scale of C values
            xlabel('omegaRScale');
            ylabel('omegaQScale');
            zlabel('C value');
            title('Consistency Measurement (C) for ' + system_name + measure_rate + prop);
            
            % Compute processed files
            processedFiles = [processedFiles, filePath];
            disp(" => Plot " + length(processedFiles) + ": " + filePath + " Finished")
        end
    end
    disp("Finish Plotting")
    disp(length(processedFiles) +" Figures generated!")

    function fileList = getAllCsvFiles(dirName)
        % Get the data for the current directory
        dirData = dir(dirName);
        
        % Find the index for directories
        dirIndex = [dirData.isdir];
        
        % Filter CSV files
        isCsvFile = cellfun(@(x) (length(x) > 4) && strcmpi(x(end-3:end), '.csv'), {dirData.name});
        fileList = {dirData(~dirIndex & isCsvFile).name}';
        
        if ~isempty(fileList)
            fileList = cellfun(@(x) fullfile(dirName, x), fileList, 'UniformOutput', false);
        end
        
        % Get a list of subdirectories excluding '.' and '..'
        subDirs = setdiff({dirData(dirIndex).name}', {'.', '..'});
        
        % Loop over valid subdirectories and call this function recursively
        for iDir = 1:length(subDirs)
            nextDir = fullfile(dirName, subDirs{iDir});
            fileList = [fileList; getAllCsvFiles(nextDir)];
        end
    end
end
