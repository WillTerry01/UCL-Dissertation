function scriptPath = getScriptPath(levelsUp)
    % GETSCRIPTPATH Return the absolute directory of the calling script.
    %
    % Syntax:
    %   scriptPath = getScriptPath();
    %   scriptPath = getScriptPath(levelsUp);
    %
    % Description:
    %   To support sandboxing of different system configurations, ebe must
    %   be able to automatically identify where various scripts live on a
    %   computer. This method returns the directory location of the script
    %   calling this method, not the location of this method itself. The
    %   levelsUp parameter lets you walk further up the call tree to see
    %   where earlier scripts are related.
    %
    %   For example, suppose you have a script test1.m located in the
    %   directory test1_tests. Calling
    %
    %     scriptPath = getScriptPath()
    %
    %   or
    %     scriptPath = getScriptPath(1)
    %
    %   will return the absolute directory test1_tests, which is where
    %   test1.m is located.
    %
    %   Suppose test1.m calls test2.m in another directory test2_tests
    %   which then calls:
    %
    %     scriptPath = getScriptPath()
    %
    %   This will return the directory test2_tests. If you instead call
    %
    %     scriptPath = getScriptPath(2)
    %
    %   it will now return the absolute directory for test1_tests.
    % 
    %   If levelsUp is greater than the size of the call stack, it returns
    %   pwd.
    %
    % Inputs:
    %   levelsUp - (int, optional)
    %      The number of levels that should be walked up. [Default: 1]
    %
    % Outputs:
    %   scriptPath - (string)
    %      The absolute location of the script found by MATLAB
    %
    % See Also:
    %   PWD

    % Default levelsUp to 1 (the immediate caller) if not provided
    if nargin < 1
        levelsUp = 1;
    end
    
    % Use dbstack to get the call stack
    stack = dbstack('-completenames');
    
    % Ensure levelsUp is within the stack range; if not, get the pwd
    if numel(stack) > levelsUp
        % Get the file of the desired level in the stack
        callingFile = stack(levelsUp + 1).file; % +1 because stack(1) is getScriptPath itself
        scriptPath = fileparts(callingFile); % Extract the path
    else
        scriptPath = pwd;
    end
end
