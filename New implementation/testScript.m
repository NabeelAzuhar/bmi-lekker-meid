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