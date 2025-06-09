function [chi2, dimX, dimZ] = ...
    runDriveBotExample(configuration, scenarioDirectory, omegaRScale, omegaQScale)

    configuration.ROdometry = omegaQScale * configuration.ROdometry;
    configuration.RGPS = omegaRScale * configuration.RGPS;

    % Set up the simulator
    simulator = drivebot.DriveBotSimulator(configuration, scenarioDirectory);
    
    % Create the localization system
    drivebotSLAMSystem = drivebot.DriveBotSLAMSystem(configuration);
    
    % Q1(e)i:
    % Use the method "setRecommendOptimizationPeriod" in DriveBotSLAMSystem
    % to control the rate at which the optimizer runs
    drivebotSLAMSystem.setRecommendOptimizationPeriod(1);
    
    % This tells the SLAM system to do a very detailed check that the input
    % appears to be correct but can make the code run slowly. Once you are
    % confident your code is working safely, you can set this to false.
    drivebotSLAMSystem.setValidateGraph(true);
    
    % Run the main loop and correct results
    results = minislam.mainLoop(simulator, drivebotSLAMSystem);

    % Return the final chi2
    chi2 = results{1}.chi2History(end);

    if (nargout > 1)
        % Calculate the dimensioin of the system
        [dimX, dimZ] = drivebotSLAMSystem.getNumOfDimensions();
    end
end