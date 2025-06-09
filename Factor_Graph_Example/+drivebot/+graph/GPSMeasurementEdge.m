classdef GPSMeasurementEdge < g2o.core.BaseUnaryEdge
   
    properties(Access = protected)
        
        xyOffset;
        
    end
    
    methods(Access = public)
    
        function this = GPSMeasurementEdge(xyOffset)
            this = this@g2o.core.BaseUnaryEdge(2);
            this.xyOffset = xyOffset;
        end
        
        function computeError(this)

	    % Q1d:
        % Implement the code
        % w[Gk] = [x[k];y[k]] + M[phi[k]]*[off[[x];off[y]] - z[Gk]
        x = this.edgeVertices{1}.estimate();
        c = cos(x(3));
        s = sin(x(3));
        M = [c -s ; s c];

        this.errorZ = x(1:2) + M * this.xyOffset - this.z;
        
%         warning('gpsmeasurementedge:computeerror:unimplemented', ...
%                 'Implement the rest of this method for Q1d.');
        end
        
        function linearizeOplus(this)

	    % Q1d:
        % Implement the code
        x = this.edgeVertices{1}.estimate();
        c = cos(x(3));
        s = sin(x(3));
        delx = this.xyOffset(1);
        dely = this.xyOffset(2);
        this.J{1} = ...
                [1 0 -delx*s-dely*c;
                0 1 delx*c-dely*s];
%         warning('gpsmeasurementedge:lineareizeoplus:unimplemented', ...
%                 'Implement the rest of this method for Q1d.');
        end
    end
end
