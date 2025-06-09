% A set of utility functions to help the operation of ebe
% Version 20250130 30-Jan-2025 
%
% Maths-Related functions 
%
%   continuousToDiscrete    - Converts a constant linear time model to a
%                             discrete time model.
%   psd_sqrtm               - Matrix square root of positive SEMI-definite
%                             matrices using eigendecompositions
%   psd_sqrtm_chol          - Matrix square root of positive SEMI-definite
%                             matrices using Cholesky decomposition
%
% File handling-related functions
% 
%   getScriptPath           - Return absolute location of script
%   prepareDirectory        - Create and / or clean up a directory
%   readJSONFile            - Recursively read JSON file
%   resolveFilePath         - Find absolute location of a file path
%
% Classes
%   PerformanceData         - A class for storing a linked list of data