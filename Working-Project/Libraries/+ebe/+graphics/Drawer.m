classdef Drawer < handle
    methods(Access = public, Abstract)
        update(obj, x, P);
    end
end