classdef ResultsAccumulator < ebe.core.Component
    % ResultsAccumulator summary of ResultsAccumulator
    %
    % Components that inherit from this class store results from the
    % estimator over time to carry out operations such as computing
    % estimation error

    properties(Access = protected)

        % The event generator in the mainloop
        eventGenerator;

        % The set of estimators which are being run
        estimators;

    end

    methods(Access = public)

        function setEventGeneratorAndEstimators(obj, eventGenerator, estimators)
            % SETEVENTGENERATORANDESTIMATORS Register the event generator
            % and set of estimators
            %
            % Syntax:
            %   resultsAccumulator.setEventGeneratorAndEstimators(eventGenerator, estimators)
            %
            % Description:
            %   Registers the event generator and the set of estimators
            %   with the results accumulator. It is assumed the event
            %   generator can sometimes provide ground truth values (e.g.,
            %   simulation), whereas the estimators produce estimates.
            %
            % Inputs:
            %   eventGenerator - (ebe.core.EventGenerator)
            %       The event generator creating the events
            %   estimators - (cell array of ebe.core.EventBasedEstimator)
            %       An array which contains the estimators which are
            %       executed in the main loop
            obj.eventGenerator = eventGenerator;
            obj.estimators = estimators;
        end
    end

    methods(Access = public, Abstract)

        collectResults(obj)
            % COLLECTRESULTS Collect the results
            %
            % Syntax:
            %   resultsAccumulator.collectResults()
            %
            % Description:
            %   Carry out any operation to collect data from the event
            %   generator and estimators to generate results. This could
            %   include, for example, getting ground truth information and
            %   computing the estimation error for each estimator.
    end
end
