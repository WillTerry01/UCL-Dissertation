classdef EventBasedEstimator < ebe.core.ConfigurableComponent
    % EventBasedEstimator: Base class for the estimator
    % 
    % This class implements the base class of an event-based estimation
    % system. Modern robotic systems are inherently multi-sensor fusion
    % systems: they take data from multiple sensors at different times.
    % As a result, there's quite a lot of management of data and timing.
    % To handle this, the estimator is given a sequencer of timestamped
    % events. It then carries out the necessary operations to manage these
    % event.

    properties(Access = public)

        % Set of registered event handlers
        eventHandlers;
        
        % Step number - how many times have we iterated?
        stepNumber;
        
        % The timestamp that the estimate refers to
        currentEstimateTime;

        % The current time
        currentTime;
                
        % Flag to show if debugging is enabled.
        debug;
        
        % Flag to show if the system has been initialized or not
        initialized;
    end

    methods(Access = public)
        function obj = EventBasedEstimator(config)
            % EventBasedEstimator Constructor for EventBasedEstimator
            %
            % Syntax:
            %   eventBasedEstimator = EventBasedEstimator(config)
            %
            % Description:
            %   Constructs an event based estimator with an empty handler map.
            %   Note this class is abstract, and cannot be directly
            %   instantiated.
            %
            % Inputs:
            %   config - (struct)
            %       The configuration structure
            %
            % Outputs:
            %   obj - (handle)
            %       An instance of a EventBasedEstimator

            obj = obj@ebe.core.ConfigurableComponent(config);
            
            % Create the handler map
            obj.eventHandlers = containers.Map();

        end

        function processEvents(obj, events)
            % PROCESSEVENTS process a time ordered set of events
            %
            % Syntax:
            %   eventBasedEstimator.processEvents(events)
            %
            % Description:
            %   This method processes incrementally works its way through
            %   the set of input events and processes them in increasing
            %   time order.
            %
            %   The estimator operates in two modes: initialized and
            %   uninitialized. When the esitmator is initialized, a state
            %   estimate has been computed. It might not be possible to do
            %   this immediately. For example, you might initialize from a
            %   set of camera measurements, and you have to obtain a
            %   sufficient number to triangulate.
            % 
            %   In the initialized mode, the estimator computes the
            %   difference between the previous event time and the current
            %   event time. If this is sufficiently large, a prediction
            %   step is run. If a handler for the named event type has
            %   been registered, the handler is called. If no handler is
            %   provided, a warning will be printed.
            % 
            %   In the unitialized mode, the prediction step cannot be
            %   carried out. Therefore, the handlers are called directly.
            %
            % Inputs:
            %   events - (cell array of ebe.core.Event objects)
            %       A time ordered set of events.
            %
            % See also:
            %   EVENT, EBE.CORE.DETAIL.ORDEREDEVENTQUEUE         
          
            % Process all the events
            for e = 1 : length(events)                
                event = events{e};

                % First check it's of the right class.
                assert(isa(event, 'ebe.core.Event'), ...
                    'eventbasedestimator:processevents:wrongobjecttype', ...
                    'The object type is %s', class(event));
                
                % Now do the actual work for this event
                obj.processEvent(event);
            end
        end

        function start(obj)
            % START Start the estimator
            %
            % Syntax:
            %   eventBasedEstimator.start()
            %
            % Description:
            %   Sets the estimator state up so that it's ready to accept
            %   the first incoming event.

            obj.currentTime = 0;
            
            % Set the current step number to zero
            obj.stepNumber = 0;
            
            % Set that we need to initialize
            obj.initialized = false;
        end
    end

    methods(Access = protected)

        function success = processEvent(obj, event)
            % PROCESSEVENT process a single event from the event queue
            %
            % Syntax:
            %   success = eventBasedEstimator.processEvent(event)
            %
            % Description:
            %   Processes the next event from the queue. The event time
            %   must be non-decreasing. See documentation on processEvents
            %   for the public API and an explanation of what's going on.
            %
            % Inputs:
            %   x - (ebe.core.Event)
            %       The event to be processed
            %
            % Outputs:
            %   success - (bool)
            %       Set to true if the event was processed successfuly,
            %       false otherwise.

            % This flag indicates if we should increment the step number
            % inside the estimator; we can only consider doing this if the
            % estimator is initialized
            incrementStepNumber = obj.initialized;

            % First see if we have to do a prediction step and, if so,
            % carry it out

            % Figure out the delta from the current estimate time
            dT = event.time() - obj.currentTime;

            % We only predict if the estimator is initialized
            if (obj.initialized == true)
    
                % Time should be non-negative otherwise there's a timestamp
                % skew.
                assert(dT >= 0);

                % If dT is less than minDT, no prediction is needed
                % otherwise, predict forwards. Each prediction step has a
                % maximum length of maxDT.
                if (dT <= obj.config.minDT)
                    obj.handleNoPrediction();
                    incrementStepNumber = false;
                else
                    predictTime = obj.currentTime;
                    while (dT > 0)
                        dTStep = min(dT, obj.config.maxDT);
                        predictTime = predictTime + dTStep;
                        obj.handlePredictForwards(dT);
                        dT = dT - dTStep;
                    end
                end
                obj.currentTime = event.time();
            end

            % If dT was large enough and we are initialized, increment the
            % step number.
            if (incrementStepNumber == true)
                % Increment the step number
                obj.stepNumber = obj.stepNumber + 1;
            end

            % Figure out if we have a regsitered handler for this event
            % type. If so, dispatch it. If not, report that there's an
            % issue and insert a handler which suppresses printing warnings
            % from this point on.
            if obj.eventHandlers.isKey(event.type())
                eventHandler = obj.eventHandlers(event.type());
                eventHandler(event);
            else
                warning('No handler registered for event type: %s', event.type());
                obj.registerEventHandler(event.type(), @obj.ignoreUnknownEventType);
            end

            % Store a step results. Note this catches the cases that either
            % (a) the estimator was just initialized or (b) the prediction
            % interval is sufficiently long that we've bumped to a new
            % time step.
            if (obj.initialized == true) && (incrementStepNumber == true)
                % Handle any post-event activities
                obj.storeStepResults();
            end

            success = true;

        end

        function registerEventHandler(obj, eventType, eventHandler)
            % REGISTEREVENTHANDLER register a handler for a named event type
            %
            % Syntax:
            %   eventBasedEstimator.registerEventHandler(eventType, eventHandler)
            %
            % Description:
            %   Registers a handler for a named eventType. When an event of
            %   type eventType is encoutered, the registered hander is
            %   called. Warning: no error checking is carried out, so it's
            %   possible to overwrite the handler.
            %
            % Inputs:
            %   eventType - (string)
            %       The event type name to be handled
            %   eventHandler - (function handle)
            %       The handle to the function that will handle the event
            %       type. This must have the signature
            %
            %         success = eventHandler(event);
            %
            %       The return value shows if the event was successfully
            %       handled or not.

            obj.eventHandlers(eventType) = eventHandler;
        end

        function success = ignoreUnknownEventType(~, ~)
            % IGNOREUNKNOWNEVENTTYPE ignores unknown event types
            %
            % Syntax:
            %   eventBasedEstimator.ignoreUnknownEventType(event)
            %
            % Description:
            %   This dummy handler is registered to suppress warnings for
            %   events of unknown types. It simply ignores the event
            %
            % Inputs:
            %   event - (ebe.core.Event)
            %       The event to be ignored
            %
            % Returns:
            %   success - (bool)
            %       Always returns true

            success = true;
        end

    end

    methods(Access = protected, Abstract)

        handleNoPrediction(obj)
            % HANDLENOPREDICTION handle the case when no prediction occurs
            %
            % Syntax:
            %   eventBasedEstimator.handleNoPrediction()
            %
            % Description:
            %   The event based estimator runs the prediction step when the
            %   estimator is initialized and the prediction step is
            %   sufficiently long. This method is called if either
            %   constraint is violated and so a prediction step is not
            %   required.

        handlePredictForwards(obj, dT)
            % HANDLEPREDICTFORWARDS handle the case when a prediction occurs
            %
            % Syntax:
            %   eventBasedEstimator.handlePredictForwards(dT)
            %
            % Description:
            %   Carry out the prediction step of duration dT
            %
            % Inputs:
            %   dT - (double)
            %       The lenght of the prediction step

        storeStepResults(obj)
            % STORESTEPRESULTS store step-by-step results
            %
            % Syntax:
            %   eventBasedEstimator.storeStepResults(dT)
            %
            % Description:
            %   Once a step has been completed (prediction + event
            %   handled), this method is called to store any data that's
            %   needed on a step-by-step basis, such as ground truth to
            %   compute estimation errors.
    end
end