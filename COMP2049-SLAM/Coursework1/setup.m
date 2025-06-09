% Script to set up the search path if not done so already
if (exist('ebe.core.Event', 'class')  ~= 8)

    % Add the path
    addpath(genpath(pwd));

    % Confirm this worked
    if (exist('ebe.core.Event', 'class')  == 8)
        disp('ebe added to the search path')
    else
        disp('ebe not added to the search path; was setup.m run from the correct directory?')
    end
else
    disp('ebe is already on the search path; type "rmpath(genpath(pwd))" on the command line if you want to clear it')
end