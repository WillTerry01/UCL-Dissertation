% This is the basic element which is stored in a hypergraph. It basically
% stores just an ID, an optional name, and information to show if it has
% been inserted into a graph.
%
% It should not be instantiated directly.


classdef HyperGraphElement < handle
    % HyperGraphElement summary of HyperGraphElement
    %
    % All classes which subclass from this one can be managed by a
    % hypergraph. The class itself mostly stores ID information (name / ID)
    % together with an indicator of which graph it's associated with and
    % whether it has been validated.
    
    properties(Access = protected)
        
        % The graph the element is regstered with
        owningGraph;
        
        % The id of the element. This is of type int64 and should be unique
        % for each element
        elementId;
        
        % The name of this object
        elementName;
        
        % Flag to show if validated
        validated;
    end
    
    methods(Access = protected)
        
        function obj = HyperGraphElement()
            % HyperGraphElement Constructor for HyperGraphElement
            %
            % Syntax:
            %   obj = HyperGraphElement()
            %
            % Description:
            %   Creates an instance of a HyperGraphElement object. Note
            %   this class contains some abstract functions and cannot be
            %   directly instantiated
            %
            % Outputs:
            %   obj - (handle)
            %       An instance of a HyperGraphElement

            obj.owningGraph = [];
            obj.elementId = -1;
            obj.elementName = '';
            obj.validated = false;
        end
        
    end
    
    methods(Access = public, Sealed = true)
        
        function name = name(obj)
            % NAME Return the name of the element.
            %
            % Syntax:
            %   name = obj.name();
            %
            % Description:
            %   Each element can have a name optionally set on it (e.g., to
            %   help with debugging). This method returns the stored name
            %   or an empty string if nothig was set.
            %
            % Outputs:
            %   name - (string)
            %       The name of the element. If not set, this returns an
            %       empty string.

            name = obj.elementName;
        end
        
        function id = id(obj)
            % ID Return the ID of the element.
            %
            % Syntax:
            %   id = obj.id();
            %
            % Description:
            %   Each element has a unique ID which is internally used to do
            %   stuff like object lookup. If registered with a graph, this
            %   has a non-negative value.
            %
            % Outputs:
            %   id - (int64)
            %       The element ID: this should be unique for a given
            %       graph.

            id = obj.elementId;
        end
        
        function graph = graph(obj)
            % GRAPH Return the instance of the graph the element is part of.
            %
            % Syntax:
            %   graph = obj.graph();
            %
            % Description:
            %   The graph the element has been registered with. Each
            %   element can be added to at most one graph. If it's not in
            %   any graph, this will return an empty structure.
            %
            % Outputs:
            %   graph - (g2o.core.HyperGraph)
            %       The graph in which this element has been registered.
            graph = obj.owningGraph;
        end
        
        function clearValidated(obj)
            % CLEARVALIDATED Clear the validation flag.
            %
            % Syntax:
            %   obj.clearValidated();
            %
            % Description:
            %   Reset the validation flag on the element. This means its
            %   validation state will be checked the next time the graph is
            %   validated.

            obj.validated = false;
        end
    end
    
    methods(Access = public, Abstract)
        
        % Make sure the element is valid. The definition of validity is
        % different for vertices and edges.
        validate(obj);
        
    end
    
    methods(Access = protected)
        
        function setId(obj, newElementId)
            % SETID Set the id of the element
            %
            % Syntax:
            %   obj.setId(newElementId);
            %
            % Description:
            %   Set the ID number of the element. This can only be done if
            %   the element is not currently sitting in any graph. An
            %   assert will fire if you try to change the ID on an element
            %   already in the graph.
            %
            % Inputs:
            %   newElementId - (int)
            %       The new element ID.
            
            % Assume we can cast okay
            newElementId = int64(newElementId);
            
            % If no ID has been assigned, we can assume the vertex hasn't
            % been registered with a graph
            if (isempty(obj.elementId) == true)
                obj.elementId = newElementId;
                return;
            end
            
            % Check we haven't been registered with a graph already
            assert((isempty(obj.owningGraph) == true), ...
                'g2o::hypergraphelement::changeidafterregistration', ...
                ['Attempt to change the id of an element that has ' ...
                'already been added to a graph; oldID=%d, newID=%d'], ...
                obj.elementId, newElementId);
            
            % Now change the vertex ID
            obj.elementId = newElementId;
            
        end
    end
    
    % These methods are only accessible by the hypergraph
    
    methods(Access = {?g2o.core.HyperGraph})
        
        function setGraph(obj, owningGraph)
            % SETGRAPH Low-level function to set the graph.
            %
            % Syntax:
            %   obj.setGraph(owningGraph);
            %
            % Description:
            %   Set the graph in which this element is placed. No error
            %   checking is carried out. This method is internally called
            %   when adding this element to a graph.
            %
            % Inputs:
            %   owningGraph - (g2o.core.HyperGraph)
            %       The graph.

            obj.owningGraph = owningGraph;
        end
        
        function clearGraph(obj)
            % CLEARGRAPH Low-level function to clear the graph.
            %
            % Syntax:
            %   obj.clearGraph();
            %
            % Description:
            %   Clears the internal graph field. This is carried out when
            %   removing an element from the graph. No error checking is
            %   carried out.
            %
            % Inputs:
            %   owningGraph - (g2o.core.HyperGraph)
            %       The graph.
            obj.owningGraph = [];
        end
    end
end