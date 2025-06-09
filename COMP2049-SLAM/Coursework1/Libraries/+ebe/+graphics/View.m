classdef View < ebe.core.ConfigurableComponent
    % View summary of View
    %
    % A view is a collection of graphical output routines which are used to
    % provide information on an event generator or estimator. A SLAM
    % system, for example, needs to depict the estimated location of the
    % platform together with all the sensors. An event generator might want
    % to show the ground truth platform position together with the sensor
    % measurements at a given timestep.

    methods(Access = public)

        function obj = View(config)
            % View Constructor for View
            %
            % Syntax:
            %   view = View(config);
            %
            % Description:
            %   Constructor for an instance of a View object. This is an
            %   abstract class and cannot be directly instantianted.
            %
            % Inputs:
            %   config - (struct)
            %       The configuration structure
            %
            % Outputs:
            %   view - (handle)
            %      An instance of a View
            obj@ebe.core.ConfigurableComponent(config);
        end

        function [handles, entries] = legendEntries(obj)
            % LEGENDENTRIES Return items that need to be posted into a
            % legend
            %
            % Syntax:
            %   [handles, entries] = view.legendEntries();
            %
            % Description:
            %   Return cell arrays of handles and entries which contain
            %   anything that needs to be posted to a figure legend.
            %

            handles = [];
            entries = {};
        end
 
    end

    methods(Access = public, Abstract)


        visualize(obj, eventArray);
            % VISUALIZE Update the visualization of this view.
            %
            % Syntax:
            %   view.visualize(events)
            %
            % Description:
            %   Carry out the steps needed to visualize this view. The
            %   current events are also fed in because some visualization
            %   actions might need them
            %
            % Inputs:
            %   events - (cell array of ebe.core.Event objects)
            %       The set of events to be visualized this time
    end
end