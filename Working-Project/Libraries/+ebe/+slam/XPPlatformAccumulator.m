classdef XPPlatformAccumulator < ebe.core.ResultsAccumulator

    % This accumulator stores the results for the platform for a SLAM
    % problem

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

            obj.timeStore = [];
            obj.xTrueStore = [];
            obj.xEstStore = cell(numel(obj.estimators), 1);
            obj.PEstStore = cell(numel(obj.estimators), 1);

        end

        function collectResults(obj)

            if (isempty(obj.timeStore) == true)
                obj.timeStore = obj.eventGenerator.time();
            else
                obj.timeStore(end + 1) = obj.eventGenerator.time();
            end

            if (isempty(obj.xTrueStore) == true)
                obj.xTrueStore = obj.eventGenerator.xTrue();
            else
                obj.xTrueStore(:, end + 1) = obj.eventGenerator.xTrue();
            end

            for e = 1 : numel(obj.estimators)
                [x, P] = obj.estimators{e}.platformEstimate();
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