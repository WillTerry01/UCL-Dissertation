classdef XPAccumulator < ebe.core.ResultsAccumulator
    % XPAccumulator summary of XPAccumulator
    %
    % An accumulator class which stores the true value from the event
    % generator, together with the mean and covariance estimates from the
    % estimators.

    properties(GetAccess = public, SetAccess = protected)

        % Time store
        timeStore;

        % Ground truth value
        xTrueStore;

        % Store of x values
        xEstStore;

        % Store of P values
        PEstStore;

    end

    methods(Access = public)

        function start(obj)
            % START Start the accumulator.
            %
            % Syntax:
            %   xpAccumulator.start()
            %
            % Description:
            %   Clear all the internal buffers and set the class ready to
            %   collect data.

            obj.timeStore = [];
            obj.xTrueStore = [];
            obj.xEstStore = cell(numel(obj.estimators), 1);
            obj.PEstStore = cell(numel(obj.estimators), 1);

        end


        function collectResults(obj)
            % COLLECTRESULTS Iterate over the event generator and
            % estimators and store all the etimates
            %
            % Syntax:
            %   xpAccumulator.collectResults()
            %
            % Description:
            %   Iterate across the event generator and estimators and store
            %   the data. If the arrays are empty, allocate them.

            % Store the time
            if (isempty(obj.timeStore) == true)
                obj.timeStore = obj.eventGenerator.time();
            else
                obj.timeStore(end + 1) = obj.eventGenerator.time();
            end

            % Store the ground truth
            if (isempty(obj.xTrueStore) == true)
                obj.xTrueStore = obj.eventGenerator.xTrue();
            else
                obj.xTrueStore(:, end + 1) = obj.eventGenerator.xTrue();
            end

            % Store the individual state estimates
            for e = 1 : numel(obj.estimators)
                [x, P] = obj.estimators{e}.computeXP();
                if (isempty(obj.xEstStore{e}))
                    obj.xEstStore{e} = x;
                    obj.PEstStore{e} = diag(P);
                else
                    obj.xEstStore{e} = cat(2, obj.xEstStore{e}, x);
                    obj.PEstStore{e} = cat(2, obj.PEstStore{e}, diag(P));
                end
            end
        end
    end
end