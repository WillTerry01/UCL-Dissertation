classdef DistinguishableColours < handle

    methods (Access=public, Static)
        function obj = getInstance()            
            persistent localInstance;
            if isempty(localInstance) || ~isvalid(localInstance)
                localInstance = ebe.graphics.DistinguishableColours();
            end
            obj = localInstance;
        end

        function colour = assignColour(name)
            if (nargin == 0)
                name = '';
            end
            colour = ebe.graphics.DistinguishableColours.getInstance().getDistinguishableColour(name);
        end

        function reset()
            ebe.graphics.DistinguishableColours.getInstance().clear();
        end
    end

    properties(Access = protected)

        % Cached values
        numCachedColours;
        cachedColours;

        % The colour which has been assigned
        coloursAssigned;

        % Set of named colours
        namedAssignedColours;

        % Should we use the old dictionary?
        useOldDictionary;
    end

    methods(Access=protected)
        
        function obj = DistinguishableColours()

            % Flag to show if we can use the new dictionary
            obj.useOldDictionary = isMATLABReleaseOlderThan("R2023b");

            % Create a cache of colours; these avoid the need to call
            % distinguishable_colors and skips one Matlab toolbox
            % dependency
            obj.cachedColours = [         0         0    1.0000
                1.0000         0         0
                     0    1.0000         0
%                     0         0    0.1724
%                1.0000    0.1034    0.7241
%                1.0000    0.8276         0
%                     0    0.3448         0
%                0.5172    0.5172    1.0000
%                0.6207    0.3103    0.2759
%                     0    1.0000    0.7586
%                     0    0.5172    0.5862
                     0         0    0.4828
                0.5862    0.8276    0.3103
%                0.9655    0.6207    0.8621
%                0.8276    0.0690    1.0000
%                0.4828    0.1034    0.4138
%                0.9655    0.0690    0.3793
%                1.0000    0.7586    0.5172
                0.1379    0.1379    0.0345
                0.5517    0.6552    0.4828
                0.9655    0.5172    0.0345
                0.5172    0.4483         0
                0.4483    0.9655    1.0000
                0.6207    0.7586    1.0000
                0.4483    0.3793    0.4828
                0.6207         0         0
                     0    0.3103    1.0000
                     0    0.2759    0.5862
                0.8276    1.0000         0
                0.7241    0.3103    0.8276
                0.2414         0    0.1034
                0.9310    1.0000    0.6897
                1.0000    0.4828    0.3793
                0.2759    1.0000    0.4828
                0.0690    0.6552    0.3793
                0.8276    0.6552    0.6552
                0.8276    0.3103    0.5172
                0.4138         0    0.7586
                0.1724    0.3793    0.2759
                     0    0.5862    0.9655
                0.0345    0.2414    0.3103
                0.6552    0.3448    0.0345
                0.4483    0.3793    0.2414
                0.0345    0.5862         0
                0.6207    0.4138    0.7241
                1.0000    1.0000    0.4483
                0.6552    0.9655    0.7931
                0.5862    0.6897    0.7241
                0.6897    0.6897    0.0345
                0.1724         0    0.3103
                     0    0.7931    1.0000
                0.3103    0.1379         0
                     0    0.7241    0.6552
                0.6207         0    0.2069
                0.3103    0.4828    0.6897
                0.1034    0.2759    0.7586
                0.3448    0.8276         0
                0.4483    0.5862    0.2069
                0.8966    0.6552    0.2069
                0.9655    0.5517    0.5862];

            obj.numCachedColours = size(obj.cachedColours, 1);

            obj.clear();
            
        end

        function colour = getDistinguishableColour(obj, name)

            % Check if the name in the map; if so, retrieve it
            if (isempty(name) == false)
                if ((isConfigured(obj.namedAssignedColours) == true) ...
                        && (isKey(obj.namedAssignedColours, name) == true))
                    colourIndex = obj.namedAssignedColours(name);
                    colour = obj.cachedColours(colourIndex, :);
                    return
                end
            end

            % Assign the colour
            obj.coloursAssigned = obj.coloursAssigned + 1;
            colour = obj.cachedColours(obj.coloursAssigned, :);

            % If a name was specified, store it in the map
            if (isempty(name) == false)
                if (obj.useOldDictionary == true)
                    obj.namedAssignedColours(name) = obj.coloursAssigned;
                else
                    obj.namedAssignedColours = insert(obj.namedAssignedColours, name, obj.coloursAssigned);
                end
            end
        end

        function clear(obj)

            if (obj.useOldDictionary == true)
                obj.namedAssignedColours = dictionary();
            else
                obj.namedAssignedColours = configureDictionary("char", "uint64");
            end
            obj.coloursAssigned = uint64(0);
        end

    end
    

end