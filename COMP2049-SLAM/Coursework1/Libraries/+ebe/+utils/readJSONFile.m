function jsonFileContent = readJSONFile(fileName, levelUp)
    % READJSONFILE Recursively read a JSON file, interpreting any field
    % which ends in ".json" with a JSON file and attempting to load that.
    %
    % Syntax:
    %   jsonFileContent = readJSONFile(fileName);
    %   jsonFileContent = readJSONFile(fileName, levelUp);
    %
    % Description:
    %   This function reads and parses a JSON file and writes the output to
    %   a MATLAB struct. If, during the read process, the function
    %   encounters a field name which contains the suffix ".json", it is
    %   assumed to be a json file and the code will attempt to load it.
    %   Note there is no error checking, so if a file loads itself it will
    %   get stuck in an infinite loop.
    %
    % Inputs:
    %   fileName - (string)
    %      The name of the JSON file to load
    %   levelUp - (int, optional)
    %      This is used by getScriptPath to work out the location of the
    %      file. It is required to support nested loading of JSON files.
    %      [Default: 0]
    %
    % Outputs:
    %   jsonFileContent - (struct)
    %      A MATLAB struct which contains the loaded JSON file. If the JSON
    %      file is nested, the child JSON files appear as nested structs.
    % See also:
    %   GETSCRIPTPATH

    
    if (nargin == 1)
        levelUp = 0;
    end
    
    jsonFileName = ebe.utils.resolveFilePath(fileName, 3 + levelUp);
    jsonFile = fileread(jsonFileName);
    
    try
        jsonFileContent = jsondecode(jsonFile);
    catch ME
        error('Failed to read or decode JSON file: %s\n%s', jsonFile, ME.message);
    end
    
    % Now loop through and find every field with the value *.json; assume this
    % is a configuration file, load it and replace the field with the file
    % content
    %
    % WARNING: This could get stuck in an infinite loop
    
    % Get all field names in the struct
    fields = fieldnames(jsonFileContent);
        
    % Copy the original struct to modify it
    updatedJSONFileContent = jsonFileContent;
        
    % Loop through each field
    for i = 1:numel(fields)
        % Get the value of the current field
        fieldValue = jsonFileContent.(fields{i});
        
        % Check if the value is a string or char array and matches '*.json'
        if (ischar(fieldValue) || isstring(fieldValue)) && endsWith(fieldValue, '.json')
            % Try to read and decode the JSON file
            try
                jsonContent = ebe.utils.readJSONFile(fieldValue, levelUp + 1);
                updatedJSONFileContent.(fields{i}) = jsonContent;
                fprintf('Replaced field "%s" with JSON-decoded content from %s\n', fields{i}, fieldValue);
            catch ME
                error('Failed to read or decode JSON file: %s\n%s', fieldValue, ME.message);
            end
        end
    end
    
    jsonFileContent = updatedJSONFileContent;
end