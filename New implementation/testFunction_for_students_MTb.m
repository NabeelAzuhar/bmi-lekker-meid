%% Continuous Position Estimator Test Script
% This script tests the continuous position estimator by decoding hand
% trajectory using pre-trained model parameters.

% Clean up
close all

% Load training data
load("monkeydata_training.mat")

% Set random number generator seed
% rng(1500);

% Shuffle trial order
ix = randperm(length(trial));

% Select training and testing data
trainingData = trial(ix(1:50), :);
testData = trial(ix(51:end), :);

fprintf('Testing the continuous position estimator...\n')

meanSquaredError = 0;
nPredictions = 0;

% Initialize figure
figure
hold on
axis square
grid

% Train Model
tic
[modelParams]= gloriaTraining(trainingData);

for trialIdx = 1:size(testData, 1)
    fprintf('Decoding block %d out of %d\n', trialIdx, size(testData, 1));
    
    % Iterate through directions
    for direction = randperm(8)
        decodedHandPos = [];
        times = 320:20:size(testData(trialIdx, direction).spikes, 2);
        
        % Iterate through time steps
        for t = times
            pastCurrentTrial.trialId = testData(trialIdx, direction).trialId;
            pastCurrentTrial.spikes = testData(trialIdx, direction).spikes(:, 1:t); 
            pastCurrentTrial.decodedHandPos = decodedHandPos;
            pastCurrentTrial.startHandPos = testData(trialIdx, direction).handPos(1:2, 1);
            
            % Decode hand position
            [decodedPosX, decodedPosY, modelParams] = estimatorTest(pastCurrentTrial, modelParams);
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            % Calculate mean squared error
            meanSquaredError = meanSquaredError + norm(testData(trialIdx, direction).handPos(1:2, t) - decodedPos)^2;
        end
        
        nPredictions = nPredictions + length(times);
        
        % Plot decoded and actual positions
        hold on
        plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r');
        plot(testData(trialIdx, direction).handPos(1, times), testData(trialIdx, direction).handPos(2, times), 'b')
    end
end
toc

% Add legend and calculate RMSE
legend('Decoded Position', 'Actual Position')
RMSE = sqrt(meanSquaredError / nPredictions);
