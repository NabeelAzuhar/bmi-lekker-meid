%% Clearing outputs and loading data
clc
clear all

load monkeydata_training.mat

duration = 1000;

%% Testing for smallest trial duration
data = trial;
for angle_num = 1:size(data, 2)
    for trial_num = 1:size(data, 1)
        spike_data = data(trial_num, angle_num).spikes;
        if size(spike_data,2) < duration
            duration = size(spike_data,2);
            low_trial = trial_num;
            low_angle = angle_num;
        end
    end
end

% Smallest duration is 571 ms in reaching angle 6 trial 71

%% Trying out LDA formatting
numAngles = 8;
numTrials = 100;
numTimeBins = 16;
labels = repmat(1:numAngles, numTrials*numTimeBins, 1);
% labelsx = labels(:);  % Ensure it's a column vector, resulting in a 12800x1 vector

%% Testing out indexing
A = [1, 2, 3; 4, 5, 6; 7, 8, 9];
subset = A(1:2, 1:2);
subset(1, 1) = 100;
disp(subset)
disp(A);  % This will output [100, 2, 3; 4, 5, 6; 7, 8, 9]





