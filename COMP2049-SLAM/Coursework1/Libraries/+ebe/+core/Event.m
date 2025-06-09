classdef Event < handle
    % Event: The class storing the events
    %
    % Instances of this class are fundamental to the operation of ebe. All
    % main sensing activities - such as initialization and receipt of
    % sensor measurements - are carried out using data contained in events.
    % Each event consists of a time, a type, and an optional payload of
    % data.

    properties(GetAccess = public, SetAccess = protected)
        
        % The time of the event (required).
        time; 
        
        % The type of the event (required).
        type;
        
        % The event data (optional)
        data;
        
        % The noise on the event data (optional symmetric PSD matrix)
        covariance;

        % Any additional information (optional)
        info;
    end

    properties(Access = public)
        % Simulator timestep number; used for debugging
        eventGeneratorStepNumber;
    end

    methods(Access = public)

        function obj = Event(time, type, data, covariance, info)
            % Event Constructor for Event
            %
            % Syntax:
            %   event = Event(time, type)
            %   event = Event(time, type, data)
            %   event = Event(time, type, data, covariance)
            %   event = Event(time, type, data, covariance, info)
            %
            % Description:
            %   Creates an instance of an Event object.
            %
            % Inputs:
            %   time - (double)
            %       The time the event is scheduled to fire
            %   data - (optional, double, can be a vector)
            %       Typically contains observation data such as z or
            %       odometry
            %   covariance - (optional, postive semidefinite square matrix)
            %       The covariance associated with the data
            %   info - (optional)
            %       Arbitrary field to contain all other data including
            %       landmark IDs when dealing with SLAM
            %
            % Outputs:
            %   obj - (handle)
            %       An instance of an Event
           
            % Copy over the common values
            obj.time = time;
            obj.type = type;

            % Return if no data / covariance
            if (nargin < 3)
                return;
            end

            % Copy the event data
            obj.data = data;

            % Return if no covariance
            if (nargin < 4)
                return
            end
            
            % If the noise is a vector, assume that it encodes a diagonal covariance
            % matrix. Therefore, reshape into a matrix. If the covariance
            % is not a vector, check that it's a square matrix
            if ((size(covariance, 1) == 1) || (size(covariance, 2) == 1))
                covariance = diag(covariance);
            else
                r = size(covariance, 1);
                c = size(covariance, 2);
                assert(r == c, ...
                    'minislam:event:covariancenotsquare', ...
                    'The covariance is non-square matrix of size (%d,%d) %s', ...
                    r, c);
            end

            % Check the covariance is positive semidefinite
            assert(min(eig(covariance)) >= 0, ...
                    'minislam:event:covariancenotpsd', ...
                    'The covariance is non-positive semidefinite');

            obj.covariance = covariance;

            if (nargin == 5)
                obj.info = info;
            end
        end
    end
end
