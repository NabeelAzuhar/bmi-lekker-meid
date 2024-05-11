ix = randperm(length(trial));
trial_split = 99;
trainingData = trial(ix(1:trial_split),:);
testData = trial(ix(trial_split:end),:);

modelParameters = positionEstimatorTraining(trainingData);

for tr=1:size(testData,1)
    for direc = 1:8
        times=320:20:size(testData(tr,direc).spikes,2);
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
%             past_current_trial.decodedHandPos = decodedHandPos;
    
            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
    
            [label] = positionEstimatorClassifications(past_current_trial, modelParameters, trial_split);
        
        end
    end
end