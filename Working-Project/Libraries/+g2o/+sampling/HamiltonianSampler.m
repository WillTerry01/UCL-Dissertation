classdef HamiltonianSampler < handle
           
    % This is an attempt to reimplement Richard's code in a more object
    % oriented way to help me understand the structure and generalise a bit
    % for some other experiments.
    
    properties(Access = protected)

        % Step size
        epsilon;
        
        % Number of Hamilton steps
        nSteps;
        
        % Enable Metropolis rejection test
        useRejectionTest;
        
        % The observation and observation covariance
        y;
        R;
        
        % The mass matrix
        M;
        
        % State and momentum and dimensions
        theta;
        p;
        ndims;

        % Number of steps used in leapfrog integrator
        newtonSteps;        
    end

    methods(Access = public)
        
        function obj = HamiltonianSampler()
            obj.nSteps = 35;
            obj.epsilon = 0.01;
            obj.useRejectionTest = true;
        end
        
        function setParameters(obj, nSteps, epsilon)
            obj.nSteps = nSteps;
            obj.epsilon = epsilon;
        end
        
        % Sample the state starting from theta0 for an observation y
        % If p0 is provided, use it at the initial momentum. Otherwise,
        % it is sampled.
        
        function [theta, acceptMove, HTrajectory, reverseTrajectory] = sample(obj, y, theta0, p0)
            
            % Store the state and the observation
            obj.y = y;
            obj.theta = theta0;
            obj.ndims = length(obj.theta);
            
            % The history of the Hamiltonians, used for debugging
            storeHTrajectory = (nargout > 2);
            if (storeHTrajectory == true)
                HTrajectory = NaN(1, obj.nSteps + 1);
            end
            
            % Measure to detect sample reversal - simple test to suggest
            % samples "folding back on themselves" so the possibility of
            % the NUTS sampler
            storeReverse = (nargout > 3);
            if (storeReverse == true)
                reverseTrajectory = NaN(2, obj.nSteps + 1);
            end
            
            % Choose the initial mass matrix
            obj.computeM();
            
            % Choose initial value of the momentum if it's not provided
            if (nargin == 4)
                obj.p = p0;
            else
                obj.p = obj.sampleP();
            end
            
            % Work out the initial Hamiltonian. This is needed for the accept
            % / reject step at the end
            H0 = obj.computeH();
            
            % Store the initial Hamiltonian if required
            if (storeHTrajectory == true)
                HTrajectory(1) = H0;
            end
            
            % Store the initial reverse value if required
            if (storeHTrajectory == true)
                reverseTrajectory(:, 1) = 0;
            end
            
            % Use the leapfrog integrator
            keepRunning = true;
            s = 1;
            
            while ((s <= obj.nSteps) && (keepRunning == true))
                % Do the leapfrog integration using (23a)-(23c)
                obj.p = obj.p - 0.5 * obj.epsilon * obj.computeDHDTheta();
                obj.theta = obj.theta + obj.epsilon * obj.computeDHDP();
                obj.p = obj.p - 0.5 * obj.epsilon * obj.computeDHDTheta();

                % Update the mass matrix; this is needed for computeDHDP.
                % It is also used in the accept / reject step so it isn't
                % wasted.
                obj.computeM();
                
                % Bump the step count
                s = s + 1;
    
                % If storing the Hamiltonian, compute the value and save it
                if (storeHTrajectory == true)
                    HTrajectory(s) = obj.computeH();
                end
                
                % If storing the reverse measure, compute the value and
                % save it
                if (storeReverse == true)
                    reverseTrajectory(1, s) = dot(obj.theta - theta0, obj.p);
                    reverseTrajectory(2, s) = reverseTrajectory(1, s)/(norm(obj.theta - theta0)*norm(obj.p));
                 end
                
                % Test for early termination goes here
                keepRunning = true;
            end
            
            % Work out the final Hamiltonian
            H = obj.computeH();
            
            % Work out the energy
            deltaH = H0 - H;
            
            % Do the sample rejection step
            %acceptMove = (obj.useRejectionTest == false) || (log(rand(1)) < deltaH);
            acceptMove = (obj.useRejectionTest == false) || (rand(1) < exp(deltaH));
            
            if (acceptMove == true)
                theta = obj.theta;
            else
                theta = theta0;
            end
        end
    end
    
    methods(Access = protected)
        
        % Sample the initial momentum
        function p0 = sampleP(obj)
            p0 = chol(obj.M)' * randn(obj.ndims, 1);
        end
    
        % Compute the Hamiltonian
        function H = computeH(obj)
            H = obj.computeV() + obj.computeT();
        end
        
        % Compute the potential energy. Assuming a Gaussian, we use the
        % function to convert theta to an observation, and then compute the
        % deltas.
        function V = computeV(obj)
            nu = obj.computeNu();
            V = 0.5 * nu' * (obj.R \ nu);
        end
        
        % Compute the kinetic energy. Assume this is Gaussian distributed
        % and is zero-mean with covariance equal to the mass matrix.
        function T = computeT(obj)
            T = 0.5 * obj.p' * (obj.M \ obj.p);
        end
        
        % Compute the Hamiltonian with respect to momentum
        function dHdP = computeDHDP(obj)
            dHdP = obj.M \ obj.p;
        end
        
        % The mass matrix; the default is that it's constant
        function computeM(obj)
            obj.M = eye(obj.ndims);
        end
        
    end
    
    methods(Access = protected, Abstract)
        
        dHDTheta = computeDHDTheta(obj);
    end    
end