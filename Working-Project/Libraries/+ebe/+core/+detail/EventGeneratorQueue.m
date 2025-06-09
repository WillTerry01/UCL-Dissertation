classdef EventGeneratorQueue < handle
    % EventGeneratorQueue summary of EventGeneratorQueue
    %
    % This class stores a time ordered series of generators which, when
    % called, will emit events in increasing time order. Instances of this
    % class are used inside the simulator.

    properties(Access = protected)

        % The set of times (in seconds) that the generators are called.
        times;

        % The list of pending generators.
        generators;
    end

    methods(Access = public)

        function obj = EventGeneratorQueue()
            % EventGeneratorQueue Constructor for EventGeneratorQueue
            %
            % Syntax:
            %   obj = EventGeneratorQueue()
            %
            % Description:
            %   Creates an instance of an EventGeneratorQueue object.
            %
            % Outputs:
            %   obj - An instance of EventGeneratorQueue

            % Construct an empty queue

            obj.clear();
        end


        function clear(obj)
            % CLEAR Empty the queue of entries
            %
            % Syntax:
            %   obj.clear()        
            %
            % Description:
            %   Deletes all entries from the queue

            obj.times = [];
            obj.generators = {};
        end

        function e = empty(obj)
            % EMPTY Determines whether the queue is empty
            %
            % Syntax:
            %   isEmpty = obj.empty()        
            %
            % Description:
            %   Determines if the queue contains any entries or not.
            %
            % Outputs:
            %   isEmpty - (bool)
            %       True if the queue is empty, false otherwise

            e = isempty(obj.times);
        end

        function insert(obj, time, eventGenerator)
            % INSERT Emplace an item in the queue
            %
            % Syntax:
            %   obj.insert(time, eventGenerator)        
            %
            % Description:
            %   Places the eventGenerator function handle on the queue with
            %   associated time time.
            %
            % Inputs:
            %   time - (double)
            %       The time the event generator will fire
            %   eventGenerator - (function handle)
            %       The handle to the function which is stored

            obj.times(end+1) = time;
            obj.generators{end+1} = eventGenerator;
            assert(length(obj.times) == length(obj.generators));

        end

        function insertWithDither(obj, earliestTime, dither, eventGenerator)
            % INSERTWITHDITHER Emplace an item in the queue with randomised
            % time
            %
            % Syntax:
            %   obj.insert(earliestTime, dither, eventGenerator)        
            %
            % Description:
            %   Places the eventGenerator function handle on the queue. The
            %   time is randomised slightly: the scheduled time is given by
            %   earliestTime + rand * dither
            %
            % Inputs:
            %   earliestTime - (double)
            %       The earliest time the event generator will fire
            %   dither - (double)
            %       The maximum delay which can happen
            %   eventGenerator - (function handle)
            %       The handle to the function which is stored

            obj.times(end+1) = earliestTime + rand * dither;
            obj.generators{end+1} = eventGenerator;
            assert(length(obj.times) == length(obj.generators));

        end

        function [time, eventGenerator] = pop(obj)
            % POP Retrieve the event generator from the queue
            %
            % Syntax:
            %   [time, eventGenerator] = obj.empty()        
            %
            % Description:
            %   Retrieves the eventGenerator with the lowest value of time
            %   and removes it from the queue. If the queue is empty, time
            %   and eventGenerator will both be empty

            % Handle case of no pending events
            if (isempty(obj.times) == true)
                time = [];
                eventGenerator = [];
                return
            end

            % Find the earliest time
            [time, idx] = min(obj.times);
            eg = obj.generators(idx);
            eventGenerator = eg{1};
            
            % Remove the items from the queue
            obj.times(idx) = [];
            obj.generators(idx) = [];
        end
    end
end