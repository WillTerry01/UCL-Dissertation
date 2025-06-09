classdef EventBasedSimulator < ebe.core.EventGenerator
    % EventBasedSimulator summary of EventBasedSimulator
    %
    % Components that inherit from this class are discrete time simulators.
    %
    % The simulator works by scheduling a series of time-ordered event
    % generators. When the step() method is called, the simulator
    % predicts (if the time step is long enough) to the next event
    % generator time, and then the next event generator is run. Repeat
    % events schedule themselves.

    properties(Access = protected)

        % The current time
        currentTime;

        % Queue which stores the next event generator which is queued up
        eventGeneratorQueue;

        % Scale which can be applied to noise. Set to 0
        % (no noise) or 1 (noise)
        noiseScale;

        % Flag; if set to false the simulator will terminate at the next
        % time step
        carryOnRunning;

        % Flag to show if debugging is enabled.
        debug;
        
        % Flag to show if the system has been initialized or not
        initialized;
    end
    
    methods(Access = public)
        
        function obj = EventBasedSimulator(config)
            % EventBasedSimulator Constructor for EventBasedSimulator
            %
            % Syntax:
            %   eventBasedSimulator = EventBasedSimulator(config)
            %
            % Description:
            %   Constructs an event based simulator using the configuration. Note
            %   this is an abstract class and cannot be instantiated
            %   directly.
            %
            % Inputs:
            %   config - (struct)
            %       The configuration structure
            %
            % Outputs:
            %   obj - (handle)
            %       An instance of an EventBasedSimulator

            obj = obj@ebe.core.EventGenerator(config);
        end

        function start(obj)
            % START Start the event generator
            %
            % Syntax:
            %   eventBasedSimulator.start()
            %
            % Description:
            %   Start the simulator so that it is ready to process the
            %   first step call.

            start@ebe.core.EventGenerator(obj);

            % Set the start time to 0.
            obj.currentTime = 0;

            % Currently set to keep running
            obj.carryOnRunning = true;
            
            % Set that we need to initialize
            obj.initialized = false;

            % Create the queue for generating the events
            obj.eventGeneratorQueue = ebe.core.detail.EventGeneratorQueue();

            % Set the noise scale
            if (obj.config.perturbWithNoise == false)
                obj.noiseScale = 0;
            else
                obj.noiseScale = 1;
            end

        end
    end

    methods(Access = public, Sealed)
        
        function T = time(obj)
            % TIME current simulator time
            %
            % Syntax:
            %   time = eventBasedSimulator.time()
            %
            % Description:
            %   Return the current time according to the simulator.
            %
            % Outputs:
            %   time - (double)
            %       The current event generator time
            T = obj.currentTime;
        end

        function carryOn =  keepRunning(obj)
            % KEEPRUNNING return a flag which indicates whether the
            % simulator should keep running or not.
            %
            % Syntax:
            %   carryOn = eventBasedSimulator.keepRunning()
            %
            % Description:
            %   The carryOn flag indicates if the simulator should be
            %   stepped again or not. It returns false if the simulation
            %   has met the maximum specified number of timesteps or some
            %   other condition has been met.
            %
            % Outputs:
            %   carryOn - (bool)
            %       True if the event generator is good to be stepped
            %       again.

            carryOn = ((obj.carryOnRunning) && ...
                (obj.stepNumber <= obj.config.maximumStepNumber));
        end

        function step(obj)
            % STEP Step the simulator a single iteration
            %
            % Syntax:
            %   eventBasedSimulator.step()
            %
            % Description:
            %   The simulator predicts to the time that the next event
            %   generator will fire and any events are collected. If
            %   the event were previously requested by the events() method,
            %   the outgoing event queue is cleared. If the events have not
            %   been requested, the new events are posted to the queue.

            if (obj.outgoingEventsDispatched == true)
                obj.outgoingEvents.clear();
            end

            % Are there any pending events? If not, assume we are done
            if (obj.eventGeneratorQueue.empty() == true)
                obj.keepRunning = false;
                return
            end

            % Get the time and event generator that's next in the queue.
            [nextEventime, generateEvent] = obj.eventGeneratorQueue.pop();

            % This flag shows if we should increment to the next step
            incrementStepNumber = obj.initialized;

            % First see if we have to do time-based prediction step. We
            % only do this if (a) the system is initialized and (b) dT is
            % sufficiently big.

            % Figure out the delta from the current estimate time
            dT = nextEventime - obj.currentTime;
            dTOrig = dT;

            % Predict only if initialized
            if (obj.initialized == true)
    
                % Time should be non-negative otherwise there's a timestamp
                % skew.
                assert(dT >= 0);

                % Do not increment if really close to the last time
                if (dT <= obj.config.minDT)
                    incrementStepNumber = false;
                end

                % Predict forwards. Make sure that each call to prediction
                % is for a duration of no more than length maxDT. Note the
                % state predictor is run even if dT <= obj.config.minDT
                predictTime = obj.currentTime;
                while (dT > obj.config.minDT)
                    dTStep = min(dT, obj.config.maxDT);
                    predictTime = predictTime + dTStep;
                    obj.handlePredictForwards(dT, predictTime);
                    dT = dT - dTStep;
                end

                % Set the current time to now
                obj.currentTime = nextEventime;
            end

             if (incrementStepNumber == true)
                obj.stepNumber = obj.stepNumber + 1;
            end

            % Generate the next event
            generateEvent();

            % Store a step results. Note this catches the cases that either
            % (a) the suimulator was just initialized or (b) the prediction
            % interval is sufficiently long that we've bumped to a new
            % time step.
            if (obj.initialized == true) && (incrementStepNumber == true)
                % Handle any post-event activities
                obj.storeStepResults();
            end

            %fprintf('Simulator: %d %f (%f)\n',obj.stepNumber, obj.currentTime, dTOrig)

        end
    end

    methods(Access = protected, Abstract)

        handlePredictForwards(obj, dT, T);
            % HANDLEPREDICTFORWARDS Carry out the time update for the
            % simulator
            %
            % Syntax:
            %   eventBasedSimulator.handlePredictForwards(dT, T)
            %
            % Description:
            %   This runs the dynamic model to estimate the state of the
            %   platform / system at the net timestep.
            %
            % Inputs:
            %   dT - (double)
            %       Length of the prediction step
            %   T - (double)
            %       The simulator time that's being predicted to

        storeStepResults(obj);
    end
end