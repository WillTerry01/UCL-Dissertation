% Package containing graphics support
% Version 20250130 30-Jan-2025 
%
% Core Classes
% 
%   DistinguishableColours  - Stores pre-assigned colours to semantic
%                             quantities such as ground truth estimates.
%   Drawer                  - Base class of drawers which generate
%                             graphical depictions for things such as
%                             ground truth positions and state estimates.
%                             The state is dynamically updated over time.
%   EstimatorView           - Wraps estimators for getting graphical
%                             output such as current state estimates.


% Specialized Drawing Classes
%   CircleDrawer            - Draws a filled circle.
% 