classdef OrderedEventQueue < handle
    % OrderedEventQueue summary of OrderedEventQueue
    %
    % This class stores a time ordered series of events. These events are
    % produced by an event generator and consumed by an estimator.

    properties(Access = protected)
        % The list of events
        eventQueue;

        % A flag to indicate whether the list needs to be sorted.
        dirty; 
    end
        
    methods(Access = public)
        
        function obj = OrderedEventQueue()
            % OrderedEventQueue Constructor for OrderedEventQueue
            %
            % Syntax:
            %   orderedEventQueue = OrderedEventQueue()
            %
            % Description:
            %   Creates an instance of a OrderedEventQueue object with an
            %   empty queue internally.
            %
            % Outputs:
            %   orderedEventQueue - An instance of OrderedEventQueue

            obj.clear();
        end
        
        function clear(obj)
            % CLEAR Empty the queue of entries
            %
            % Syntax:
            %   orderedEventQueue.clear()        
            %
            % Description:
            %   Deletes all entries from the queue

            obj.eventQueue = {};
            obj.dirty = false;
        end
        
        function insert(obj, newEvents)
            % INSERT Emplace one or more events in the queue
            %
            % Syntax:
            %   orderedEventQueue.insert(newEvents)        
            %
            % Description:
            %   Place newEvents into the queue
            %
            % Inputs:
            %   newEvents - (ebe.core.Event or cell array of ebe.core.Event)
            %       The event or set of events to store

            % Flag as dirty
            obj.dirty = true;
            
            % Handle the case of a single event first
            if (iscell(newEvents) == false)
                obj.eventQueue{end+1} = newEvents;
                return
            end
            
            % Now handle the case of a cell array
            for c = 1 : length(newEvents)
                obj.eventQueue{end+1} = newEvents{c};
            end
        end
        
        function sortedEvents = events(obj)
            % EVENTS Sort and return cell array of events
            %
            % Syntax:
            %   sortedEvents = orderedEventQueue.events()        
            %
            % Description:
            %   Sort all the stored events and in ascending time order and
            %   return a cell array from it.
            %
            % Inputs:
            %   sortedEvents - (cell array of ebe.core.Event)
            %       The event or set of events to store

            % If the queue is dirty, it needs to be sorted
            if (obj.dirty == true)
                times = cellfun(@(e) e.time, obj.eventQueue);
                [~, idx] = sort(times, 'ascend');
                obj.eventQueue = obj.eventQueue(idx);
                obj.dirty = false;
            end

            % Return the sorted event queue
            sortedEvents = obj.eventQueue;
        end        
    end
end
