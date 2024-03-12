clear all;
clc;
%% Import Data
% 100 trials for each of 8 reaching angles.
% There are 98 neurons in each trial and the time bin is 1ms, with 0
% denoting no spike and 1 denoting spike.
load('monkeydata_training.mat');

%% Raster plot of reach 1 trial 1
r1t1 = trial(1,1).spikes;
% Create a raster plot
[num_events, num_samples] = size(r1t1);
[events, samples] = find(r1t1);

% Plot each event occurrence as a dot in the raster plot
figure;
hold on;
plot(samples, events, '.', 'MarkerSize', 5);
xlabel('Time (ms)');
ylabel('Neuron number');
title('Raster Plot for Reach 1 Trial 1');
ylim([0, num_events + 1]); % Adjust ylim to include all events
hold off;

%% Raster plot of one neuron unit over many trials
figure;
colors = {[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 0.5, 0]};

for reach = 1:8
    subplot(2, 4, reach);
    hold on;
    for trials = 1:100
%         row_num = ((reach-1)*100)+trials;
        n1 = trial(trials,reach).spikes(1,:);
        [events, samples] = find(n1);
        plot(samples, trials, '.', 'Color', colors{reach}, 'MarkerSize', 5);
    end
    title(['Reach ', num2str(reach)]);
    xlabel('Time (ms)');
    ylabel('Trial number');
end

%% PSTH


