classdef Component < handle
    % Component: The base class for ebe objects
    %
    % Many ebe objects (estimators, event generators, graphic drawing
    % routines) have a lifecycle to them: they have to start and stop to
    % make sure everything happens in the correct sequence. This base class
    % provides interfaces to do that

    properties(Access = protected)

        % The name of this component (optional)
        componentName;

    end

    methods(Access = public)            
        function obj = Component()
            % Component Constructor for Component
            %
            % Syntax:
            %   obj = Component()
            %
            % Description:
            %   Constructor for an instance of a Component object. Note
            %   that Component itself is abstract, and so cannot be
            %   directly instantiated.
            %
            % Outputs:
            %   obj - (handle)
            %      An instance of a Component
        end
    end

    methods(Access = public, Sealed)
        
        function setName(obj, name)
            % SETNAME Set the name of the component
            %
            % Syntax:
            %   obj.setName(name)        
            %
            % Description:
            %   Sets the objet name to name. Used for output purposes.
            %
            % Inputs:
            %   name - (string)
            %       The name of the component

            obj.componentName = name;
        end

        function n = name(obj)
            % NAME Retrieve the name of the component
            %
            % Syntax:
            %   componentName = obj.name();        
            %
            % Description:
            %   Returns the name of the component. Empty if not set.
            %
            % Outputs:
            %   name - (string)
            %       The name of the component
            n = obj.componentName;
        end

    end

    methods(Access = public)

        function stop(obj)
            % STOP Stop the component
            %
            % Syntax:
            %   obj.stop();        
            %
            % Description:
            %   Carries out any procedure for stopping the component. This
            %   could include carrying out any pending changes and closing
            %   log files.
        end

    end

    methods(Access = public, Abstract)

        start(obj);
            % START Start the componenent.
            %
            % Syntax:
            %   obj.start();        
            %
            % Description:
            %   Carries out any procedure for starting the component. Note
            %   that this implementation is abstract, whereas for the
            %   related method stop it is not. The reason is that
            %   non-trivial start operations are required in almost all
            %   cases but the stop operation is rarely used.
        
    end

end