% This class implements an event-based estimation system using g2o and
% the barebones for building up a minimal, ideal SLAM system. The system is
% event-based and responds to a sequence of events which are time stamped
% and served in order. To implement your SLAM system, you will need to
% implement various methods which mostly involve working with the graph.
% These methods are initially stubbed out and will generate exceptions if
% you try to call them.

classdef DriveBotSLAMSystem < minislam.slam.SLAMSystem
    
    properties(Access = public, Constant)
        % Platform state dimension
        NP = 3;
        
        % Landmark dimension
        NL = 2;
        
        % Initial cache size; might help a bit with performance
        INITIAL_CACHE_SIZE = 10000;
    end
    
    properties(Access = protected)
        
        % The most recently created vehicle vertex.
        currentVehicleVertex;
        
        % The set of all vertices associated with the vehicle state over
        % time.
        vehicleVertices;
        vehicleVertexId;
        
        % The set of all prediction edges. These are removed from the graph
        % afterwards if we don't use prediction
        processModelEdges;
        numProcessModelEdges;
        
        % The landmark vertices. Confusingly enough, "Map" here refers to
        % the data structure which is used to store the landmarks. (It
        % allows random access of landmarkID to landmark object.)
        landmarkIDStateVectorMap;
        
        % How often we recommend running the optimization
        recommendOptimizationPeriod;
        
        % Flag to show if we should prune the edges. This is needed for
        % question Q3a
        removePredictionEdgesFromGraph;
        keepFirstPredictionEdge;

        % Q3b -- attribute that when set to be true calls for graph pruning
        % to take place 
        graphPrune = false;
        
        % landmark ids
        landmarkIds

        % landmark P
        P_landmark
    end
    
    methods(Access = public)
        
        % Create the localization system and start it up.
        function this = DriveBotSLAMSystem(configuration)
            
            % Call the base class constructor
            this = this@minislam.slam.SLAMSystem(configuration);
            
            % Preallocate for convenience
            this.vehicleVertices = cell(1, this.INITIAL_CACHE_SIZE);
            
            % No vehicle vertices initally set
            this.vehicleVertexId = 0;
            
            % The set of prediction edges, initially empty
            this.processModelEdges = cell(1, this.INITIAL_CACHE_SIZE);
            this.numProcessModelEdges = 0;
            
            % Allocate the landmark map
            this.landmarkIDStateVectorMap = containers.Map('KeyType', 'int64', 'ValueType', 'any');
            
            % By default, run very infrequently
            this.recommendOptimizationPeriod = inf;
            
            this.removePredictionEdgesFromGraph = false;
            this.keepFirstPredictionEdge = false;
        end
        
        % Destroy the graph when we destroy the SLAM system.
        % Without this, MATLAB will crash whenever this object is destroyed.

        function delete(this)
            vertices = this.graph.vertices();

            for v = 1 : length(vertices)
                this.graph.removeVertex(vertices{v});
            end
        end
        
        % Recommend if an optimization is a good idea. Based on an event,
        % some activities (e.g., such as loop closing) can have a very big
        % impact on the estimates. The logic we have here just recommends
        % an optimization if a fixed number of steps have been completed.
        
        function recommendation = recommendOptimization(this)
            
            % This is how to do it after every 100 steps
            recommendation = rem(this.stepNumber, ...
                this.recommendOptimizationPeriod) == 0;
        end
        
        % Set the value of how often recommend optimization should return
        % true
        function setRecommendOptimizationPeriod(this, newRecommendOptimizationPeriod)
            this.recommendOptimizationPeriod = newRecommendOptimizationPeriod;
        end
        
        % Return the current mean and covariance estimate of the robot.
        % This is only valid after optimization has been called.
        function [x, P] = platformEstimate(this)
            [xS, PS] = this.graph.computeMarginals(this.currentVehicleVertex);
            x=full(xS);
            P=full(PS);
        end
        
        % Returns the entire history of the platform estimates. Suppose
        % there are n vehicle vertices. T is a 1 by N dimensional vector of
        % timesteps. X is a 3 by N dimensional vector of vehicle state (x,
        % y, theta). P is a 3 by N dimensional vector where the nth column
        % are the diagonals from the covariance matrix.
        function [T, X, P] = platformEstimateHistory(this)
            
            % Extract the graph
            [xS, PS] = this.graph.computeMarginals();
            
            % Create the output array
            X = zeros(this.NP, this.vehicleVertexId);
            P = zeros(this.NP, this.vehicleVertexId);
            T = zeros(1, this.vehicleVertexId);
            
            % Copy the outputs over
            for v = 1 : this.vehicleVertexId
                idx = this.vehicleVertices{v}.hessianIndex();
                
                T(v) = this.vehicleVertices{v}.time();
                
                % Copy the estimate into the array. If the vertices is
                % fixed (conditioned), its estimate is okay. The covariance
                % is not explicitly defined, but has a value of zero.
                % Therefore we fill this manually.
                if (isempty(idx) == true)
                    X(:, v) = this.vehicleVertices{v}.estimate();
                    P(:, v) = zeros(3, 1);
                else
                    X(:, v) = full(xS(idx));
                    P(:, v) = full(diag(PS(idx, idx)));
                end
            end
        end
        
        % Return the means and covariances of the landmark estimates. These
        % are only valid after optimization has been called.
        function [x, P, landmarkIds] = landmarkEstimates(this)
            
            landmarkVertices = values(this.landmarkIDStateVectorMap);
            
            numberOfLandmarks = length(landmarkVertices);
            
            landmarkIds = NaN(1, numberOfLandmarks);
            x = NaN(this.NL, numberOfLandmarks);
            P = NaN(this.NL, this.NL, numberOfLandmarks);
            
            [xS, PS] = this.graph.computeMarginals();
            
            for l = 1 : numberOfLandmarks
                landmarkIds(l) = landmarkVertices{l}.landmarkId();
                idx = landmarkVertices{l}.hessianIndex();
                x(:, l) = full(xS(idx));
                if (isempty(idx == true))
                    P(:, :, l) = zeros(3, 3);
                else
                    P(:, :, l) = full(PS(idx, idx));
                end
            end
        end
        
        % We overload the optimize method so that you can add additional
        % logic here
        function chi2 = optimize(this, maximumNumberOfOptimizationSteps)
            
            % Remove the prediction edges if requested.
            if (this.removePredictionEdgesFromGraph == true)
                this.deleteVehiclePredictionEdges();
            end

            % graph prune if requested.
            if (this.graphPrune == true)
                this.graphPruning()
            end
            
            % Now call the actual optimizer. Let it handle the default if
            % no steps are specified.
            if (nargin > 1)
                chi2 = optimize@minislam.slam.SLAMSystem(this, ...
                    maximumNumberOfOptimizationSteps);
            else
                chi2 = optimize@minislam.slam.SLAMSystem(this);
            end
            
            % if pruning, then compute the covariance for the landmarks after the opimization             
            if (this.graphPrune == true)
                [x_landmark, this.P_landmark, this.landmarkIds] = landmarkEstimates(this);
            end
        end
        
        function setRemovePredictionEdges(this, removeEdges, keepFirst)
            this.removePredictionEdgesFromGraph = removeEdges;
            this.keepFirstPredictionEdge = keepFirst;
            
        end
        
        % Q3b -- function to set the attribute of graph pruning 
        function setGraphPruning(this, toPruneOrNot)
            this.graphPrune = toPruneOrNot;
        end 

        function [dimX, dimZ] = getNumOfDimensions(this)
            dimX = 0;
            dimZ = 0;
            edges = this.graph.edges();
            vertices = this.graph.vertices();
    
            for e = 1 : length(edges)
                dimZ = dimZ + edges{e}.dimension();
            end
    
            for v =  1 : length(vertices)
                dimX = dimX + vertices{v}.dimension();
            end
        end
    end
    
    % These are the methods you will need to overload
    methods(Access = protected)
        
        % Handle the initial condition
        
        function handleInitialConditionEvent(this, event)
            
            % Create the first vertex, set its estimate to the initial
            % value and add it to the graph.
            this.currentVehicleVertex = drivebot.graph.VehicleStateVertex(this.currentTime);
            this.currentVehicleVertex.setEstimate(event.data);
            this.graph.addVertex(this.currentVehicleVertex);
            
            % Set the book keeping for this initial vertex.
            this.vehicleVertexId = 1;
            this.vehicleVertices{this.vehicleVertexId} = this.currentVehicleVertex;
            
            % If the covariance is 0, the vertex is known perfectly and so
            % we set it as fixed. If the covariance is non-zero, add a
            % unary initial prior condition edge instead. This adds a soft
            % constraint on where the state can be.
            if (det(event.covariance) < 1e-6)
                this.currentVehicleVertex.setFixed(true);
            else
                initialPriorEdge = drivebot.graph.InitialPriorEdge();
                initialPriorEdge.setMeasurement(event.data);
                initialPriorEdge.setInformation(inv(event.covariance));
                initialPriorEdge.setVertex(this.currentVehicleVertex);
                this.graph.addEdge(initialPriorEdge);
            end
        end
        
        function handleNoPrediction(~)
            % Nothing to do
        end
        
        function handleHeartbeatEvent(this, ~)
            % Nothing to do
        end
        
        function handlePredictToTime(this, time, dT)

            % Create the next vehicle vertex and add it to the graph
            
            this.currentVehicleVertex = drivebot.graph.VehicleStateVertex(time);
            
            % Q1b:
            % Implement prediction code here
            % warning('drivebotslam:handlepredicttotime:unimplemented', ...
            % 'Implement the rest of this method for Q1b.');

            % Initialise the edge and set time interval dT
            e = drivebot.graph.VehicleKinematicsEdge(dT);
            
            % Set two vertices, the first one is x[k], the second one is x[k+1]
            e.setVertex(1, this.vehicleVertices{this.vehicleVertexId});
            e.setVertex(2, this.currentVehicleVertex); 
            
            % Set the measurement u[k]
            e.setMeasurement(this.u);

            % Set the inverse of the covariance, v[k]
            e.setInformation(inv(this.uCov));
            e.initialize();

            % adding edge to graph
            this.graph.addEdge(e);

            % adding the current vehicle vertex to the graph
            this.graph.addVertex(this.currentVehicleVertex);
            
            % Increment the edge number
            this.numProcessModelEdges = this.numProcessModelEdges + 1;
            this.processModelEdges{this.numProcessModelEdges} = e;

            % Bump the indices
            this.vehicleVertexId = this.vehicleVertexId + 1;
            this.vehicleVertices{this.vehicleVertexId} = this.currentVehicleVertex;
        end
        
        function handleGPSObservationEvent(this, event)

            % Q1d:
            % Create a GPS measurement edge and add it to the graph
            % warning('drivebotslam:handlegpsobservationevent:unimplemented', ...
            % 'Implement the rest of this method for Q1c.');

            % Create a GPS measurement edge
            xyOffset = this.configuration.gpsPositionOffset;
            e = drivebot.graph.GPSMeasurementEdge(xyOffset);

            % Link current vertex
            e.setVertex(1, this.currentVehicleVertex);

            % Set measurement matrix and measurement covariance
            e.setMeasurement(event.data);
            e.setInformation(inv(event.covariance));

            % Add edge to the graph
            this.graph.addEdge(e);

        end
        
        function handleCompassObservationEvent(this, event)
            
            % Create a compass measurement edge and add it to the graph
            compassMeasurementEdge = drivebot.graph.CompassMeasurementEdge(this.configuration.compassAngularOffset);
            compassMeasurementEdge.setVertex(1, this.currentVehicleVertex);
            compassMeasurementEdge.setMeasurement(event.data);
            compassMeasurementEdge.setInformation(inv(event.covariance));
            this.graph.addEdge(compassMeasurementEdge);
        end
        
        function handleLandmarkObservationEvent(this, event)
            
            % Iterate over all the landmark measurements
            for l = 1 : length(event.landmarkIds)
                
                % Get the landmark vertex associated with this measurement.
                % If necessary, a new landmark vertex is created and added
                % to the graph.
                [landmarkVertex, newVertexCreated] = this.createOrGetLandmark(event.landmarkIds(l));
                z = event.data(:, l);

                % Q2b:
                % Complete the implementation


                % Initialise the Landmark Range Bearing Edge
                landmarkRangeBearingEdge=drivebot.graph.LandmarkRangeBearingEdge();

                % Connect current vertex with the landmark vertex
                landmarkRangeBearingEdge.setVertex(1,this.currentVehicleVertex);
                landmarkRangeBearingEdge.setVertex(2, landmarkVertex);

                % Set measurement matrix and measurement covariance
                landmarkRangeBearingEdge.setMeasurement(z);
                landmarkRangeBearingEdge.setInformation(inv(event.covariance));

                % Initialize if new landmark vertex created
                if (newVertexCreated == true)
                    landmarkRangeBearingEdge.initialize();
                end

                % Add edge to graph
                this.graph.addEdge(landmarkRangeBearingEdge);
            end
        end

        function deleteVehiclePredictionEdges(this)

            % Q3a:            
            % find the number of edges 
            edges = this.graph.edges();
            num_Edges = length(edges);

            % initialize the edge counter
            edge_count=0;
            % loop around all the edges and delete the process edge 
            for i = 1:num_Edges
                
                % check if it's a prediction edge 
                if class(edges{i}) == "drivebot.graph.VehicleKinematicsEdge"
                    edge_count=edge_count+1;
                    
                    % check if we need to keep the first edge 
                    if (this.keepFirstPredictionEdge == true && edge_count == 1)
                        % if it is the first edge, do nothing
                        continue 
                    else 
                       % otherwise, remove the current edges 
                        this.graph.removeEdge(edges{i});
                    end 
                end  
            end 
        end
        
        function graphPruning(this)            
            % Q3b:
            % implement the pruning

            % Get the number of vertices in current graph
            vertices = this.graph.vertices();
            numVertices = length(vertices);   

            % Get the number of edges in current graph
            edges=this.graph.edges();
            numEdges=length(edges);



            % set a threshold for the maximum number of edges that a landmark can have
            Threshold = round(numVertices/50)+70;


            for i = 1:numVertices % loop through all vertices 
                vertex = vertices{i};

                % Check if the current vertex is landmark state vertex
                if class(vertex) == "drivebot.graph.LandmarkStateVertex"
                    % get all the edges of this vertex
                    vertexEdges = vertex.edges();
                    numVertexEdges = length(vertexEdges);

                % find the index of covariance for a landmark vertex
                index_to_p=find(this.landmarkIds==vertex.landmarkId);

                % if the covariance exist for this landmark then adjust its threshold due to its cov
                if (~isempty(index_to_p))
                    detCov=abs(det(this.P_landmark(index_to_p)));
                    Threshold=Threshold*(-0.1+exp(sqrt(detCov)));
                end
                    % remove all the edges that beyond the threshold edge number for a landmark
                    if numVertexEdges > Threshold
                        for j = Threshold:numVertexEdges 
                            this.graph.removeEdge(vertexEdges{round(j)});
                        end
                    end
                end 
            end
            
        numEdges
        length(this.graph.edges())
        end 
        
        % This method returns a landmark associated with landmarkId. If a
        % landmark exists already, it is returned. If it does not exist, a
        % vertex is created and is added to the graph.
        function [landmarkVertex, newVertexCreated] = createOrGetLandmark(this, landmarkId)

            
            % If the landmark exists already, return it
            if (isKey(this.landmarkIDStateVectorMap, landmarkId) == true)
                landmarkVertex = this.landmarkIDStateVectorMap(landmarkId);
                newVertexCreated = false;
                return
            end
            
            fprintf('Creating landmark %d\n', landmarkId);
            
            % Create the new landmark add it to the graph
            landmarkVertex = drivebot.graph.LandmarkStateVertex(landmarkId);
            this.landmarkIDStateVectorMap(landmarkId) = landmarkVertex;
            
            this.graph.addVertex(landmarkVertex);
            
            newVertexCreated = true;
        end
        
        function storeStepResults(this)
            % Nothing
        end
        
    end
end
