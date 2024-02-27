%%% Team Members: Nabeel Azuhar Mohammed, Gloria Sun, Ioana Lazar, Alexia
%%% Badea
%%% BMI Spring 2015 (Update 17th March 2015)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         PLEASE READ BELOW            %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function positionEstimator has to return the x and y coordinates of the
% monkey's hand position for each trial using only data up to that moment
% in time.
% You are free to use the whole trials for training the classifier.

% To evaluate performance we require from you two functions:

% A training function named "positionEstimatorTraining" which takes as
% input the entire (not subsampled) training data set and which returns a
% structure containing the parameters for the positionEstimator function:
% function modelParameters = positionEstimatorTraining(training_data)
% A predictor named "positionEstimator" which takes as input the data
% starting at 1ms and UP TO the timepoint at which you are asked to
% decode the hand position and the model parameters given by your training
% function:

% function [x y] = postitionEstimator(test_data, modelParameters)
% This function will be called iteratively starting with the neuronal data 
% going from 1 to 320 ms, then up to 340ms, 360ms, etc. until 100ms before 
% the end of trial.


% Place the positionEstimator.m and positionEstimatorTraining.m into a
% folder that is named with your official team name.

% Make sure that the output contains only the x and y coordinates of the
% monkey's hand.

clc;
clear all;

load('monkeydata_training.mat');

%% Test

[explained, score] = positionEstimatorTraining(trial);
selected_scores = score(:, 1:20);

%%

% figure;
% hold on;
% % plot(1:size(rate{1,1}, 2), rate{1,1}(1,:), 'DisplayName', 'Rate');
% % plot(1:size(norm{1,1}, 2), norm{1,1}(1,:), 'DisplayName', 'Normed');
% % plot(1:size(fil{1,1}, 2), fil{1,1}(1,:), 'DisplayName', 'Filtered');
% hold off;
% grid on;
% legend();

%% Functions

function [explained, score] = positionEstimatorTraining(trial)
  % Arguments:
  
  % - training_data:
  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  
  % ... train your model
  
  % Return Value:
  
  % - modelParameters:
  %     single structure containing all the learned parameters of your
  %     model and which can be used by the "positionEstimator" function.

  % Number of trials and reaching angles
  num_trials = size(trial, 1);
  num_angles = size(trial, 2);
  num_neurons = size(trial(1,1).spikes, 1);
  bin_size = 30;
  filtered_database = cell(num_trials, num_angles);
%   rate_database = cell(num_trials, num_angles);
%   normalised_database = cell(num_trials, num_angles);



  % Preprocessing
  for n = 1 : num_trials
      for k = 1 : num_angles
          spike_data = trial(n,k).spikes;
          total_time = size(spike_data, 2);
          num_bins = ceil(total_time / bin_size);
          
          % Generating time bins
          bin_temp = zeros(num_neurons, num_bins);
          for i = 1 : num_neurons
              for j = 1: num_bins
                  bin_start = (j - 1) * bin_size + 1;
                  bin_end = min(j * bin_size, total_time);
                  bin_temp(i, j) = sum(spike_data(i, bin_start : bin_end));
              end
          end
          
          rate_temp = (bin_temp / bin_size) * 1000;
          
          % Normalise time bin spikes across neurons
%           rate_database{n, k} = rate_temp;
          rate_temp_z = zscore(rate_temp, 0, 2);
%           normalised_database{n, k} = rate_temp_z;
          
          % Moving average to smoothen the data
          window_size = 5;
          b = (1 / window_size) * ones(1, window_size);
          a = 1;
          filtered_data = zeros(size(rate_temp_z));
          for i = 1 : size(rate_temp_z, 1)
              filtered_data(i, :) = filter(b, a, rate_temp_z(i, :));
          end

          filtered_database{n, k} = filtered_data;
      end
  end

  % Principle Component Analysis (PCA)
  total_time_bins = 0;
  for n = 1 : num_trials
      for k = 1 : num_angles
          total_time_bins = total_time_bins + size(filtered_database{n, k}, 2);
      end
  end

  total_time_bins

  pca_array = zeros(total_time_bins, num_neurons);
  current_row = 1;
  for n = 1 : num_trials
      for k = 1 : num_angles
          current_bin = size(filtered_database{n, k}, 2);
          pca_array(current_row : current_row + current_bin - 1, :) = filtered_database{n, k}';
          current_row = current_row + current_bin;
      end
  end

  [coeff, score, latent, tsquared, explained] = pca(pca_array);




end

function [x, y] = positionEstimator(test_data, modelParameters)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

  % - test_data:
  %     test_data(m).trialID
  %         unique trial ID
  %     test_data(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     test_data(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by your
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times your function has been called previously on
  %         the same data sequence.
  %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 320):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = []
  %             test_data.spikes = 98x320 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = [2.3; 1.5]
  %             test_data.spikes = 98x340 matrix of spiking activity
  
  
  
  % ... compute position at the given timestep.
  
  % Return Value:
  
  % - [x, y]:
  %     current position of the hand
   
end

