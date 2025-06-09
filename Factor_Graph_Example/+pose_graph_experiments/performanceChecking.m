meanCovariance_processBias = load("D:\University\UCL\project\week17\mean_Covariance_0.1_2.csv");
meanMSE_processBias = load("D:\University\UCL\project\week17\mean_MSE_0.1_2.csv");
% meanError = load("D:\University\UCL\project\week17\mean_Error_0.1_2.csv");

meanCovariance_observationBias = load("D:\University\UCL\project\week17\mean_Covariance_2_0.1.csv");
meanMSE_observationBias = load("D:\University\UCL\project\week17\mean_MSE_2_0.1.csv");
meanCovariance_optimal = load("D:\University\UCL\project\week17\mean_Covariance_1_1.csv");
meanMSE_optimal = load("D:\University\UCL\project\week17\mean_MSE_1_1.csv");
meanCovariance_bayes = load("D:\University\UCL\project\week17\mean_Covariance_0.96273_1.0016.csv");
meanMSE_bayes = load("D:\University\UCL\project\week17\mean_MSE_0.96273_1.0016.csv");

gps_meanCovariance_processBias = load("D:\University\UCL\project\week17\gps_mean_Covariance_0.1_2.csv");
gps_meanMSE_processBias = load("D:\University\UCL\project\week17\gps_mean_MSE_0.1_2.csv");
% meanError = load("D:\University\UCL\project\week17\mean_Error_0.1_2.csv");

gps_meanCovariance_observationBias = load("D:\University\UCL\project\week17\gps_mean_Covariance_2_0.1.csv");
gps_meanMSE_observationBias = load("D:\University\UCL\project\week17\gps_mean_MSE_2_0.1.csv");
gps_meanCovariance_optimal = load("D:\University\UCL\project\week17\gps_mean_Covariance_1_1.csv");
gps_meanMSE_optimal = load("D:\University\UCL\project\week17\gps_mean_MSE_1_1.csv");
gps_meanCovariance_bayes = load("D:\University\UCL\project\week17\gps_mean_Covariance_0.98672_0.97918.csv");
gps_meanMSE_bayes = load("D:\University\UCL\project\week17\gps_mean_MSE_0.98672_0.97918.csv");



% % Plotting
% figure;
% subplot(2,2,1);
% hold off
% plot(meanError(1,:), 'r')
% timestepMeanX = mean(meanError(1,:), 1);
% hold on
% plot(movmean(timestepMeanX, 200), 'LineWidth', 4, 'Color', 'cyan')
% title('X');
% xlabel('Time Step');
% ylabel('Value');
% 
% subplot(2,2,2);
% hold off
% plot(meanError(2,:), 'g')
% timestepMeanXdot = mean(meanError(2,:), 1);
% hold on
% plot(movmean(timestepMeanXdot, 200), 'LineWidth', 4, 'Color', 'cyan')
% title('X dot');
% xlabel('Time Step');
% ylabel('Value');
% 
% subplot(2,2,3);
% hold off
% plot(meanError(3,:), 'b')
% timestepMeanY = mean(meanError(3,:), 1);
% hold on
% plot(movmean(timestepMeanY, 200), 'LineWidth', 4, 'Color', 'cyan')
% title('Y');
% xlabel('Time Step');
% ylabel('Value');
% 
% subplot(2,2,4);
% hold off
% plot(meanError(4,:), 'black')
% timestepMeanY = mean(meanError(4,:), 1);
% hold on
% plot(movmean(timestepMeanY, 200), 'LineWidth', 4, 'Color', 'cyan')
% title('Y dot');
% xlabel('Time Step');
% ylabel('Value');

figure;
% subplot(2,1,1);
hold on;
plot(meanMSE_optimal, 'Color', 'r', 'LineWidth', 3);
plot(meanMSE_bayes, 'Color', 'g', 'LineWidth', 2);
plot(meanMSE_observationBias, 'Color', 'b', 'LineWidth', 3);
plot(meanMSE_processBias, 'Color', 'cyan', 'LineWidth', 3);

legend('Optimal', 'GPBO', 'Observation-bias', 'Process-bias');
title('State Estimation Mean Squared Error');
xlabel('Time Step');
ylabel('Value');
hold off;

figure
% subplot(2,1,2);
hold on;
plot(meanCovariance_optimal, 'Color', 'r', 'LineWidth', 3);
plot(meanCovariance_bayes, 'Color', 'g', 'LineWidth', 2);
plot(meanCovariance_observationBias, 'Color', 'b', 'LineWidth', 3);
plot(meanCovariance_processBias, 'Color', 'cyan', 'LineWidth', 3);

legend('Optimal', 'GPBO', 'Observation-bias', 'Process-bias');
title('Mean Covariance (Diagonal Average)');
xlabel('Time Step');
ylabel('Value');
hold off;


figure;
% subplot(2,1,1);
hold on;
plot(gps_meanMSE_optimal, 'Color', 'r', 'LineWidth', 3);
plot(gps_meanMSE_bayes, 'Color', 'g', 'LineWidth', 2);
plot(gps_meanMSE_observationBias, 'Color', 'b', 'LineWidth', 3);
plot(gps_meanMSE_processBias, 'Color', 'cyan', 'LineWidth', 3);

legend('Optimal', 'GPBO', 'Observation-bias', 'Process-bias');
title('State Estimation Mean Squared Error');
xlabel('Time Step');
ylabel('Value');
hold off;

figure
% subplot(2,1,2);
hold on;
plot(gps_meanCovariance_optimal, 'Color', 'r', 'LineWidth', 3);
plot(gps_meanCovariance_bayes, 'Color', 'g', 'LineWidth', 2);
plot(gps_meanCovariance_observationBias, 'Color', 'b', 'LineWidth', 3);
plot(gps_meanCovariance_processBias, 'Color', 'cyan', 'LineWidth', 3);

legend('Optimal', 'GPBO', 'Observation-bias', 'Process-bias');
title('Mean Covariance (Diagonal Average)');
xlabel('Time Step');
ylabel('Value');
hold off;

% % Plotting
% figure;  % Create a new figure
% hold on;  % Hold on to add multiple plots to the same figure
% 
% plot(meanMSE, 'Color', 'r', 'LineWidth', 3);          % Plot mean data from X in red
% plot(meanCovariance, 'Color', 'b', 'LineWidth', 3);  % Plot mean data from trueX in blue
% 
% % Add legend and labels
% legend('Mean MSE', 'Mean Covariance');
% xlabel('Timesteps');
% ylabel('Value');
% 
% hold off;  % Release the hold on the figure