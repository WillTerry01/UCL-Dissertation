classdef ConfigurableComponent < ebe.core.Component
    % ConfigurableComponent: Components which take a configuration
    % structure
    %
    % Some ebe objects can be configured using a configuration struct. This
    % class extends the component to store the configuration.

    properties(Access = protected)

        % The stored configuration struct
        config;

    end

    methods(Access = public)

        function obj = ConfigurableComponent(config)
            % ConfigurableComponent Constructor for ConfigurableComponent
            %
            % Syntax:
            %   obj = ConfigurableComponent(config)
            %
            % Description:
            %   Constructs a configurable component which takes a config
            %   struct input. Note this is an abstract class, and cannot be
            %   directly instantiated.
            %
            % Inputs:
            %   config - (struct)
            %       The configuration structure
            %
            % Outputs:
            %   obj - (handle)
            %       An instance of a ConfigurableComponent

            obj = obj@ebe.core.Component();
            obj.config = config;
        end
    end

end