%% Initialisations

clear all;
close all;
clc;

% load trial data
load('monkeydata_training.mat');

%% Visualise data structure

% 100 trials for 8 reaching angles, each is 1 struct with 3 fields
% 3 fields: trialID, spikes, handPos
% The indices k = 1, . . ., 8 correspond to the reaching angles
% (30/180π, 70/180π, 110/180π, 150/180π, 190/180π,230/180π, 310/180π, 350/180π) respectively

% visualise 1st reaching angle (x100 trials)
angle1 = trial(:,1);
spikes1 = angle1.spikes; % 98 units over 672 time stamps (in ms)
tracj1 = angle1.handPos; % 3 positions over 672 time stamps (in ms)

%% raster plot for a single trial (trial 1,1)
% uses spike data
raster_data = trial(1,1).spikes; % 1 trial over 8 reaching angles
[neuron, t] = find(raster_data);

scatter(t, neuron, 10, 'filled');
title('Raster Plot for Trial (1,1)');
xlabel('time (ms)');
ylabel('neuron number');

%% raster plot for 1 unit over many trials
unit = 1; % take the first unit
y_axis = 1:100;
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'o'];
angle = 1;

figure(1)
for n = 1:100 % loop through each trial
    raster_data = trial(n, angle).spikes(unit,:);
    [neuron, t] = find(raster_data);
    scatter(t, y_axis(n), 10, colors(angle), 'filled');
    hold on
end
hold off

titleString = sprintf('Raster Plot for unit %d, angle %d', unit, angle);
title(titleString);
xlabel('time (ms)');
ylabel('trial number');

%% Plot hand positions for different trials in x, y plane
n = 1; % trial number

figure(2)
for angle = 1:8
    data = trial(n, angle).handPos;
    x = data(1, :);
    y = data(2, :);
    plot(x, y,'LineWidth', 1);
    grid on
    grid minor
    hold on
end
hold off

legend('ang.1', 'ang.2', 'ang.3', 'ang.4', 'ang.5', 'ang.6', 'ang.7', 'ang.8');
titleString = sprintf('Hand Trajectories for trial %d across 8 angles', n);
title(titleString);
xlabel('x position');
ylabel('y position');

%% Tuning Curve

% Number of neurons
num_neurons = size(trial(1,1).spikes, 1);
% Number of directions
num_directions = size(trial, 2);

% Preallocate arrays to hold firing rates and standard deviations
firing_rates = zeros(num_neurons, num_directions);
std_devs = zeros(num_neurons, num_directions);

% Calculate firing rates for each neuron and direction
for neuron = 1:num_neurons
    for direction = 1:num_directions

        % Collect all spike data for this neuron and direction across all trials
        all_spikes = [];

        for n = 1:size(trial, 1) % loop through 100 trials
            spikes = trial(n, direction).spikes(neuron, :);
            all_spikes = [all_spikes spikes];
        end
        
        % Calculate the firing rate: number of spikes divided by total time
        % Adjust the time calculation based on your time frame
        firing_rate = sum(all_spikes) / (size(all_spikes, 2));
        firing_rates(neuron, direction) = firing_rate;
        
        % Calculate the standard deviation of firing rates for this neuron and direction
        trial_rates = zeros(1, size(trial, 1));
        for n = 1:size(trial, 1) % loop through 100 trials
            trial_spikes = trial(n, direction).spikes(neuron, :);
            trial_rate = sum(trial_spikes) / (length(trial_spikes));
            trial_rates(n) = trial_rate;
        end
        std_devs(neuron, direction) = std(trial_rates);
    end
end

%% Plot tuning curves for a given neuron with error bars
neuron = 1;
figure; % Open a new figure for each neuron
errorbar(1:num_directions, firing_rates(neuron, :), std_devs(neuron, :));
title(sprintf('Tuning Curve for Neuron %d', neuron));
xlabel('Movement Direction');
ylabel('Firing Rate (spikes/s)');
grid on
