% Installation script

mexFileName = 'sparseinv_mex';

% See if we need to build the sparseinv mex file, if so, set it up
if (exist(mexFileName, 'file') == 3)
    fprintf('Found sparseinv mexfile at %s\n', which('sparseinv_mex'));

else
    disp('Building sparseinv mex file');
    sparseInvFile = which('sparseinv.m');
    [filepath,~,~] = fileparts(sparseInvFile);
    currentPwd = pwd;
    cd(filepath);
    sparseinv_install
    cd(currentPwd);
end

% Test if the MEX file loads properly
try
%    feval('sparseinv_mex'); % Try calling the MEX file without arguments
    feval(mexFileName)
catch ME
    % If the error indicates architecture mismatch
    if contains(ME.message, 'Invalid MEX file')
        error('MEX file "%s" is incompatible with your architecture. Please rebuild it', mexFileName);
    else
        fprintf('MEX file "%s" appears to load correctly\n', mexFileName);
    end
end
