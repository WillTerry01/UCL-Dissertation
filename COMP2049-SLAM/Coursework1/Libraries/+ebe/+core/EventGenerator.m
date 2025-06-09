classdef EventGenerator < ebe.core.ConfigurableComponent
    % EventGenerator summary of EventGenerator
    %
    % Components that inherit from this class are sources of events,
    % including simulators and log file readers.
    %
    % As the event generator progresses, it builds up a queue of outgoing
    % events that need to be handled. When the set of events is requested,
    % the internal event queue is emptied and is gradually filled again.

    properties(Access = protected)

        % The step number
        stepNumber;

        % The most recent events
        outgoingEvents;

        % 
        outgoingEventsDispatched;
    end
    
    methods(Access = public)
        
        function obj = EventGenerator(config)
            % EventGenerator Constructor for EventGenerator
            %
            % Syntax:
            %   eventGenerator = EventGenerator(config)
            %
            % Description:
            %   Constructs an event generator using the configuration. Note
            %   this is an abstract class and cannot be instantiated
            %   directly.
            %
            % Inputs:
            %   config - (struct)
            %       The configuration structure
            %
            % Outputs:
            %   eventGenerator - (handle)
            %       An instance of an EventGenerator
            obj = obj@ebe.core.ConfigurableComponent(config);
        end

        function start(obj)
            % START Start the event generator
            %
            % Syntax:
            %   eventGenerator.start()
            %
            % Description:
            %   Sets the estimator state up so that it's ready to generate
            %   events. This could include, for example, opening log files
            %   for reading.
           
            % Set the current step number to zero
            obj.stepNumber = 0;

            % Set up new event queue for outgoing events
            obj.outgoingEvents = ebe.core.detail.OrderedEventQueue();

            % Flag that we have not dispatched any pending events
            obj.outgoingEventsDispatched = false;
        end

    end

    methods(Access = public, Sealed)
        
        function nextEvents = events(obj)
            % EVENTS get any outgoing events that need handling
            %
            % Syntax:
            %   nextEvents = eventGenerator.events()
            %
            % Description:
            %   Retrieve any unhandled events that have been scheduled. An
            %   internal flag is also set which instructs the generator to
            %   clear the list of outgoing events the next time step is
            %   called.
            %
            % Outputs:
            %   nextEvents - (cell array of ebe.core.Event objects)
            %       The set of events. These are sorted in time increasing
            %       order.
            nextEvents = obj.outgoingEvents.events();
            obj.outgoingEventsDispatched = true;
        end

        function clearOutgoingEvents(obj)
            % CLEAROUTGOINGEVENTS clears the list of outgoing events
            %
            % Syntax:
            %   eventGenerator.clearOutgoingEvents()
            %
            % Description:
            %   Empty the outgoing event queue

            obj.outgoingEvents.clear();
        end
        
        function stepNumber = stepCount(obj)
            % STEPCOUNT returns the step count in the generator
            %
            % Syntax:
            %   stepCount = eventGenerator.stepCount()
            %
            % Description:
            %   Return the count of how many times the event generator has
            %   stepped forwards. Note that this is not the same as the
            %   number of times step() is called. 
            %
            % Outputs:
            %   stepCount - (int)
            %       The number of steps which have been taken.
            
            stepNumber = obj.stepNumber;
        end
    end

    methods(Access = public, Abstract = true)

        carryOn = keepRunning(obj);
            % KEEPRUNNING return a flag to show if the mainloop should keep
            % running or not
            %
            % Syntax:
            %   carryOn = eventGenerator.keepRunning()
            %
            % Description:
            %   Typically an event generator runs for a fixed amount of
            %   time before finishing (e.g., because it's reached the end
            %   of a log file, timeout reached). This flag returns if the
            %   generator is "done" or not.
            %
            % Outputs:
            %   carryOn - (bool)
            %       Flag to indicate if the main loop should keep running
            %       or not.

        step(obj);
            % STEP Step the event generator
            %
            % Syntax:
            %   eventGenerator.step()
            %
            % Description:
            %   The generator operates in a finite series of timesteps.
            %   This method causes the next step to be taken. This might be
            %   to simulate the next activity, or read in the next line of
            %   a log file.
        
        T = time(obj)
            % TIME current event generator time
            %
            % Syntax:
            %   time = eventBasedEstimator.time()
            %
            % Description:
            %   Internally, the event generator keeps a track of time. This
            %   returns the current value of it
            %
            % Outputs:
            %   time - (double)
            %       The current event generator time
        
    end
end