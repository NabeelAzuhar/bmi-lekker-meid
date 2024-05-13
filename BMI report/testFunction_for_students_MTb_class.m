% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

function RMSE = testFunction_for_students_MTb_class(teamName)

load monkeydata_training.mat

% Set random number generator
rng(2013);
ix = randperm(length(trial));

% addpath(teamName);

% Select training and testing data (you can choose to split your data in a different way if you wish)
trial_split = 60;
trainingData = trial(ix(1:trial_split),:);
testData = trial(ix(trial_split:end),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  
actualClassificationLabels = [];
classifiedLabels = [];

figure
hold on
axis square
grid

tic;
% Train Model
modelParameters = positionEstimatorTrainingClassification(trainingData);

for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            actualClassificationLabels = [actualClassificationLabels, direc];

            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimatorClassifications') == 4
                [decodedPosX, decodedPosY, newParameters, label] = positionEstimatorClassifications(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimatorClassifications') == 3
                [decodedPosX, decodedPosY, label] = positionEstimatorClassifications(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            classifiedLabels = [classifiedLabels, label];
            
        end
        n_predictions = n_predictions+length(times);
        hold on
%         plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
%         plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end
end

elapsedTime = toc;

% legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions);
classificationSuccessRate = 100 - ((sum(actualClassificationLabels ~= classifiedLabels) / size(actualClassificationLabels, 2)) * 100);

fprintf('RMSE: %.2f\n', RMSE)
fprintf('Time taken: %.2f\n', elapsedTime)
fprintf('Classification success rate: %.2f%%\n', classificationSuccessRate)

% rmpath(genpath(teamName))

end
