% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

function [RMSE] = testFunction_for_students_MTb()

load monkeydata_training.mat

% Set random number generator
rng(2013);
ix = randperm(length(trial));

% addpath(teamName);

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  

figure
hold on
axis square
grid

tic;
% Train Model
[modelParameters] = gloriaTraining_goo(trainingData);

for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8)
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('estimatorTest_goo') == 3
                [decodedPosX, decodedPosY, newParameters] = estimatorTest_goo(past_current_trial, modelParameters);
%                 if newParameters.actualLabel ~= direc
%                     disp(t)
%                     disp(newParameters.actualLabel)
%                 end
                modelParameters = newParameters;
            elseif nargout('estimatorTest_goo') == 2
                [decodedPosX, decodedPosY] = estimatorTest_goo(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            
        end
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end
end

legend('Decoded Position', 'Actual Position')

timeElapsed = toc;

RMSE = sqrt(meanSqError/n_predictions);



% rmpath(genpath(teamName))
fprintf('RMSE: %.4f\n', RMSE);
fprintf('Time elapsed: %.4f seconds\n', timeElapsed);
end