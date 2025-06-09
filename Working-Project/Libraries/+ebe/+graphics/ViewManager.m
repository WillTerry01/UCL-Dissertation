classdef ViewManager < ebe.core.ConfigurableComponent

    % Class to manage views contained in a single figure

    properties(Access = protected)

        % Handle for the figure
        figureHandle;

        % The set of views
        views;

    end

    methods(Access = public)

        function obj = ViewManager(config)
            obj@ebe.core.ConfigurableComponent(config);
            obj.views = {};
        end

        function addView(obj, view)
            obj.views{end+1} = view;
        end

        function start(obj)
            % Start everything, pulling legends as we go
            legendHandles = [];
            legendEntries = {};
            for v = 1 : numel(obj.views)
                obj.views{v}.start();
                [handles, entries] = obj.views{v}.legendEntries();
                if (isempty(entries) == false)
                    legendHandles(end + 1) = handles;
                    legendEntries{end + 1} = entries;
                end
            end
            legend(legendHandles, legendEntries);
        end

        function stop(obj)
            for v = 1 : numel(obj.views)
                obj.views{v}.stop();
            end
        end

        function visualize(obj, eventArray)
            for v = 1 : numel(obj.views)
                obj.views{v}.visualize(eventArray);
            end
        end
    end
end