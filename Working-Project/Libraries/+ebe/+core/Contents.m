% Package containing core components of the ebe library
% Version 20250130 30-Jan-2025 
%
% Classes 
%   Component               - Base-level class of all components
%                             (event generators, graphics, etc.)
%   ConfigurableComponent   - A Component object which takes a
%                             configuration structure to control its
%                             behaviour.
%   Event                   - Packages up type, time and data.
%   EventBasedEstimator     - Takes in a sequence of events and performs
%                             some estimation action such as localization
%                             or SLAM. Note that since different algorithms
%                             can be implemented, they are not defined here
%   EventBasedSimulator     - Base class to support discrete time event
%                             simulation.
%   EventGenerator          - Base class of any discrete time event
%                             generator system. Subclasses include the
%                             EventBasedSimulator but also anything reading
%                             a log file.
%   ResultsAccumulator      - Collects output from the event generator and
%                             estimators to compute things like estimation
%                             errors.
% 
% Namespaces
%   ebe.core.detail         - Some internal utility classes used to support
%                             core components.
