classdef SaveFrameAction < ebe.graphics.PostDrawAction

    properties(Access = protected)

        fileTemplate;
        figureName;
        frameCount;
        figureState;
    end

    methods(Access = public)

        function obj = SaveFrameAction(fileTemplate, figureName)
            obj.fileTemplate = fileTemplate;
            obj.figureName = figureName;
        end

        function start(obj)

            % Create the directory if necessary
            ebe.utils.prepareDirectory(obj.fileTemplate);

            % Clear the frame count
            obj.frameCount = 0;

            obj.figureState = ebe.graphics.FigureManager.getFigure(obj.figureName, true);

        end

        function run(obj)

            obj.frameCount = obj.frameCount + 1;

            frame = getframe(obj.figureState.figureHandle);

            fileName = sprintf(obj.fileTemplate, obj.frameCount);

            imwrite(frame.cdata, fileName);
        end

    end

end