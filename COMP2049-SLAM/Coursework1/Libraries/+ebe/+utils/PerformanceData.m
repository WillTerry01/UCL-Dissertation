classdef PerformanceData < handle
    % PerformanceData summary of PerformanceData
    %
    % Store which holds various kinds of performance data, such as time
    % requried and number of dimensions. Data is posted using a key and a
    % value pair. Each key refers to an array which dynamically grows in
    % time. Numerical quantities are stored in vectors or matrices.
    % Non-numeric quantities are stored in cell arrays. The order o the
    % arrya preserves the order of insertion.
    %
    % Written with aid from GPT.

    properties (Access = private)
        dictionary;
    end

    methods(Access = public)

        function obj = PerformanceData()
            % PerformanceData Constructor for PerformanceData
            %
            % Syntax:
            %   performanceData = PerformanceData();
            %
            % Description:
            %   Creates an instance of a PerformanceData object.

            obj.dictionary = configureDictionary('string', 'cell');

        end

        function obj = add(obj, key, value)
            % ADD Record data
            %
            % Syntax:
            %   performanceData.add(key, value);
            %
            % Description:
            %   Store the data in value in the key named key. Each key
            %   stores a cell array. All numerical data is in a 
            %
            % Inputs:
            %   key - (string)
            %       The key where the data should be stored.
            %   value - (numeric or non-numeric type)
            %       The value which has to be stored.

            % First check if the value exists. If it doesn't, create the
            % cell array and insert the value
            try
                cellValue = lookup(obj.dictionary, key);
            catch
                obj.dictionary = insert(obj.dictionary, key, {value});
                return
            end

            % Retrieve the value stored inside the cell
            existingValue = cellValue{1};

            % If the cell is numeric, append numerical quantities if we are
            % trying to add a numeric value
            if isnumeric(existingValue)
                if (isnumeric(value) == true)
                    obj.dictionary = insert(obj.dictionary, key, {[existingValue; value]});
                    return
                end
                error('Attempt to add non-numeric performance value to numeric performance value.');
            end
            
            % Non-numeric case: Append to a growing cell array
            existingValue{end+1} = value;
            obj.dictionary = insert(obj.dictionary, key, {existingValue});
        end
        
        function value = get(obj, key)
            cellValue = lookup(obj.dictionary, key);
            value = cellValue{1};
        end
        
        function display(obj)
            % Custom display function to show stored data
            disp(obj.dictionary);
        end
    end
end