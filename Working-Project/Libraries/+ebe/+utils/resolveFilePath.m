function fullPath = resolveFilePath(filePath, levelsUp)
    % RESOLVEFILEPATH Find the absolute directory of a file
    %
    % Syntax:
    %   fullPath = resolveFilePath(filePath);
    %   fullPath = resolveFilePath(filePath, levelsUp);
    %
    % Description:
    %   This function takes a file path and tries to find its absolute
    %   location on a computer. If the filePath appears to be absolute,
    %   then it is simply returned. If it is relative, it is assumed to be
    %   relative to a calling script, identifed by levelsUp.
    %
    % Inputs:
    %   filePath - (string)
    %      The file path to the file that's being searched for.
    %   levelsUp - (int, optional)
    %      The number of levels walked up the tree. [Default: 2]
    %
    % See also:
    %   GETSCRIPTPATH

    if (nargin == 0)
        levelsUp = 2;
    end

    % First convert fileName to have the correct file separation for the
    % platform type
    filePath = strrep(filePath, '/', filesep);

    % Check if the given file path is absolute
    if isabsolute(filePath)
        % If it's absolute, just return it as-is
        fullPath = filePath;
    else
        % If it's relative, construct the absolute path using getScriptPath
        callingScriptPath = ebe.utils.getScriptPath(levelsUp);
        fullPath = fullfile(callingScriptPath, filePath); % Combine paths
    end
end

% OS-independent way to figure out if this is an absolute path or not
function absFlag = isabsolute(path)
    % Check if the path is absolute
    if ispc % Windows
        absFlag = ~isempty(regexp(path, '^[a-zA-Z]:\\', 'once')); % Drive letter followed by '\'
    else % Mac/Linux
        absFlag = startsWith(path, '/'); % Absolute paths start with '/'
    end
end
