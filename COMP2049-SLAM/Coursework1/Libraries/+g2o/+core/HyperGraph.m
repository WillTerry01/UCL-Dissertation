% The base class which is used to describe the estimation problems. These
% are encoded as a hypergraph which consists of vertices and edges. The
% "hyper" comes from the fact that, unlike a conventaional graph in which
% edges only connect two vertices together, the edges can connect any
% number of vertices together.

classdef HyperGraph < handle
    
    properties(Access = protected)
        
        % The vertices stored in the graph
        verticesMap;
        
        % The edges stored in the graph
        edgesMap;
        
    end
    
    methods(Access = public)
       
        function obj = HyperGraph()
            % Create a new empty hypergraph instance
            
            % Create the containers to hold the vertices and edges
            obj.verticesMap = containers.Map('KeyType', 'int64', 'ValueType', 'any');
            obj.edgesMap = containers.Map('KeyType', 'int64', 'ValueType', 'any');
            
        end
        
        function N = numVertices(obj)
            N = obj.verticesMap.Count;
        end

        function vertices = vertices(obj)
            % Return a cell array which is the list of vertices added to the graph.
            vertices = values(obj.verticesMap);
        end
        
        function M = numEdges(obj)
            M = obj.edgesMap.Count;
        end

        function edges = edges(obj)
            % Return a cell array which is the list of edges added to the graph.
            edges = values(obj.edgesMap);
        end

        function obj = addVertex(obj, vertex)
            
            % Add a new vertex to the graph.
            
            % Check the vertex is of the right kind
            assert(isa(vertex, 'g2o.core.BaseVertex'), 'g2o:hypergraph:addvertex:wrongclass', ...
                'The object should inherit from base.Vertex; the object class is %s', ...
                class(vertex));
            
            % Check if the vertex has an assigned ID; if not fail,
            vertexId = vertex.id();
            assert(isempty(vertexId) == false, 'g2o:hypergraph:addvertex:unassignedid', ...
                'The vertex Id is not assigned');
            
            % Check if a vertex with this ID has already been registered;
            % if so, fail
            assert (isKey(obj.verticesMap, vertexId) == false, 'g2o:hypergraph:addvertex:repeatid', ...
                'Attempt to register a vertex with a duplicate ID %d', vertexId);
            
            % Check if the vertex has already been registered with a graph.
            % This catches the case where the vertex has been assigned to a
            % different graph already
            assert((isempty(vertex.graph()) == true), ...
                'g2o:hypergraph:addvertex:alreadyregistered', ...
                'The vertex has already been registered with a graph');
            
            % Now insert the vertex
            obj.verticesMap(vertexId) = vertex;
            
            % Flag as inserted
            vertex.setGraph(obj);
            
        end
        
        function removeVertex(obj, vertex)
            
            % Remove a vertex from the graph. When the vertex is removed, it will remove itself from all edges where it is used. Edges with no vertices are removed. 
        
            % Check the vertex is of the right kind
            assert(isa(vertex, 'g2o.core.BaseVertex'), 'g2o:hypergraph:removevertex:wrongclass', ...
                'The object should inherit from base.Vertex; the object class is %s', ...
                class(vertex));
            
            % Check if the vertex has an assigned ID; if not fail,
            vertexId = vertex.id();
            assert(isempty(vertexId) == false, 'g2o:hypergraph:removevertex:unassignedid', ...
                'The vertex Id is not assigned');
            
            % Check if a vertex with this ID has already been registered;
            % if not, fail
            assert (isKey(obj.verticesMap, vertexId) == true, 'g2o:hypergraph:removevertex:repeatid', ...
                'Attempt to remove a vertex with ID %d which has not be registered with this graph', vertexId);
            
            % Check if the vertex has already been registered with a graph.
            % This catches the case where the vertex has been assigned to a
            % different graph already
            assert(vertex.graph() == obj, ...
                'g2o:hypergraph:removevertex:alreadyregistered', ...
                'The vertex is registered with a different graph');

            % Deregister the vertex
            vertex.clearGraph();
            
            % Now go through all the edges with the vertex and remove this
            % vertex from them. If an edge has no vertices left, then
            % schedule to remove it
            edgesToRemove = {};
            edges = vertex.edges();
            for e = 1 : length(edges)
                edges{e}.removeVertex(vertex);
                if (edges{e}.numberOfUndefinedVertices() == edges{e}.numberOfVertices())
                    edgesToRemove{length(edgesToRemove)+1} = edges{e};
                end
            end
            
            % Now remove the orphaned edges
            for e = 1 : length(edgesToRemove)
                edgesToRemove{e}.clearGraph();
                remove(obj.edgesMap, edgesToRemove{e}.id());
            end
            
            % Now remove the vertex
            remove(obj.verticesMap, vertexId);
        end
        
        function addEdge(obj, edge)
            
            % Add a new edge to the graph.
            
            % Check the edge is of the right kind
            assert(isa(edge, 'g2o.core.BaseEdge'), 'g2o:hypergraph:addedge:wrongclass', ...
                'The object should inherit from base.Edge; the object class is %s', ...
                class(edge));
            
            % Check if the edge has an assigned ID; if not fail,
            edgeId = edge.id();
            assert(isempty(edgeId) == false, 'g2o:hypergraph:addedge:unassignedid', ...
                'The edge Id is not assigned');
            
            % Check if a edge with this ID has already been registered;
            % if so, fail
            assert (isKey(obj.edgesMap, edgeId) == false, 'g2o:hypergraph:addedge:repeatid', ...
                'Attempt to register an edge with a duplicate ID %d', edgeId);
            
            % Check if the edge has already been registered with a graph.
            % This catches the case where the edge has been assigned to a
            % different graph already.
            assert((isempty(edge.graph()) == true), ...
                'g2o:hypergraph:addedge:alreadyregistered', ...
                'The edge has already been registered with a graph');
            
            % Now insert the edge into the graph
            obj.edgesMap(edgeId) = edge;
            edge.setGraph(obj);
            
        end
        
        % Remove an edge from the graph
         function removeEdge(obj, edge)
             
             % Remove an edge from the graph. When an edge is removed, it removes itself from all of its vertices. Vertices with no edges are removed. 

            % Check the edge is of the right kind
            assert(isa(edge, 'g2o.core.BaseEdge'), 'g2o:hypergraph:removeedge:wrongclass', ...
                'The object should inherit from base.Edge; the object class is %s', ...
                class(edge));
            
            % Check if the edge has an assigned ID; if not fail,
            edgeId = edge.id();
            assert(isempty(edgeId) == false, 'g2o:hypergraph:removeedge:unassignedid', ...
                'The edge Id is not assigned');
            
            % Check if a edge with this ID has already been registered;
            % if not, fail
            assert (isKey(obj.edgesMap, edgeId) == true, 'g2o:hypergraph:removeedge:repeatid', ...
                'Attempt to remove an edge with ID %d which has not be registered with this graph', edgeId);
            
            % Check if the edge has already been registered with a graph.
            % This catches the case where the edge has been assigned to a
            % different graph already.
            assert(edge.graph() == obj, ...
                'g2o:hypergraph:removeedge:alreadyregistered', ...
                'The edge has not been registered with a graph');
            
            % Now go through all the edges with the vertex and remove this
            % vertex from them. If a vertes has no edges left, then
            % schedule to remove it
            verticesToRemove = {};
            vertices = edge.vertices();
            for v = 1 : length(vertices)
                vertices{v}.removeEdge(edge);
                if (vertices{v}.numberOfEdges() == 0)
                    verticesToRemove{length(verticesToRemove)+1} = vertices{v};
                end
            end
            
            % Now remove the orphaned vertices
            for v = 1 : length(verticesToRemove)
                verticesToRemove{v}.clearGraph();
                remove(obj.verticesMap, verticesToRemove{v}.id());
            end
                        
            % Deregister the edge from the graph
            edge.clearGraph();

            % Now remove the edge
            remove(obj.edgesMap, edgeId);
         end

         % Destructor to clean the graph up
         function delete(obj)
            obj.verticesMap = {};
            obj.edgesMap = {};
         end

    end
end

