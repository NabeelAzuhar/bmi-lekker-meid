%% raster plot for single trial
load('monkeydata_training.mat');
direction = 1;
trial_no = 1;
spikes_per_trial = trial(trial_no,direction).spikes;
trialid = trial(trial_no,direction).trialId;
% plot raster per trial

[neurons_idx, timebin_idx] = find(spikes_per_trial);
figure();
scatter(timebin_idx, neurons_idx, 10, 'k', 'filled'); % Adjust marker size (10) as needed
xlabel('Time bins (ms)',"Fontsize",16);
ylabel('Unit number',"Fontsize",16);
title(['Raster plot for trial ', num2str(trialid)],"Fontsize",18);

%% raster plot for one neural unit over many trials

figure()
neuron_no = 1;
color = ['r','b','g','k','y','m','c','k'];
trial_number = 1;
for direction=1:8
    for trial_no=1:50
        spikes_per_neuron = trial(trial_no,direction).spikes(neuron_no,:);
        timebin_idx = find(spikes_per_neuron);
        scatter(timebin_idx, trial_number, 10, color(direction), 'filled');
        hold on
        trial_number=trial_number+1;
    end
    

end
xlabel('Time bins (ms)',"Fontsize",16);
ylabel('Trial number',"Fontsize",16);
title(['Raster plot for neuron ', num2str(neuron_no)],"Fontsize",18)
legend("direction 1","direction 2","direction 3","direction 4","direction 5","direction 6","direction 7","direction 8")

%% PSTH

%% hand trajectories
figure()
color = ['r','b','g','k','y','m','c','k'];
for direction=1:8
    for trial_no=1:100
        trajectory = trial(trial_no,direction).handPos(1:2,:);
        plot(trajectory(1,:),trajectory(2,:),"Color", color(direction));
        hold on
    end
   
end

xlabel('X position',"Fontsize",16);
ylabel('Y position',"Fontsize",16);
title("Hand trajectories","Fontsize",18)

%% Tuning Curve
% firing rate across time and trials as a function of movement direction



