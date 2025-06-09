classdef PostDrawAction < ebe.core.Component
    % PostDrawAction summary of PostDrawAction
    % A postdraw action is called after the graphics have been updated.
  
    methods(Access = public, Abstract = true)
        
        run(obj);
            % RUN Run the action.
            %
            % Syntax:
            %   postDrawAction.run();        
            %
            % Description:
            %   Executes the registered code.

        
    end
    
end