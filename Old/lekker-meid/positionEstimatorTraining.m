% clc;
% clear all;

%% Test
% load('monkeydata_training.mat');
% [params] = positionEstimatorTraining(trial);

%%

% figure;
% hold on;
% % plot(1:size(rate{1,1}, 2), rate{1,1}(1,:), 'DisplayName', 'Rate');
% % plot(1:size(norm{1,1}, 2), norm{1,1}(1,:), 'DisplayName', 'Normed');
% % plot(1:size(fil{1,1}, 2), fil{1,1}(1,:), 'DisplayName', 'Filtered');
% hold off;
% grid on;
% legend();

% figure;
% hold on;
% cumvar = cumsum(explained);
% plot(cumvar, '-o');
% yline(80, 'r--');
% hold off;

%% Functions

function [modelParameters] = positionEstimatorTraining(trainingData)
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

  % Initialising data
  num_trials = size(trainingData, 1);
  num_angles = size(trainingData, 2);
  num_neurons = size(trainingData(1,1).spikes, 1);
  num_axis = 2;
  bin_size = 30;
  filtered_database = cell(num_trials, num_angles);
  hand_pos_database = cell(num_trials, num_angles);
%   rate_database = cell(num_trials, num_angles);
%   normalised_database = cell(num_trials, num_angles);

  % Preprocessing
  for n = 1 : num_trials
      for k = 1 : num_angles
          spike_data = trainingData(n,k).spikes;
          hand_data = trainingData(n,k).handPos(1:2, :);
          total_time = size(spike_data, 2);
          num_bins = ceil(total_time / bin_size);
          
          % Generating time bins
          bin_temp = zeros(num_neurons, num_bins);
          hand_temp = zeros(num_axis, num_bins);
          % Generating time bins for spikes
          for i = 1 : num_neurons
              for j = 1: num_bins
                  bin_start = (j - 1) * bin_size + 1;
                  bin_end = min(j * bin_size, total_time);
                  bin_temp(i, j) = sum(spike_data(i, bin_start : bin_end));
              end
          end
          
          % Converting spike count to spike rate
          rate_temp = (bin_temp / bin_size) * 1000;

          % Generating time bins for hand position
          for i = 1 : num_axis
              for j = 1 :num_bins
                  bin_start = (j - 1) * bin_size + 1;
                  bin_end = min(j * bin_size, total_time);
                  hand_temp(i, j) = mean(hand_data(i, bin_start : bin_end), 'all');
              end
          end
          
          % Normalise
          rate_temp_z = zscore(rate_temp, 0, 2);
          hand_temp_z = zscore(hand_temp, 0, 2);
          
          % Moving average to smoothen the spike   data
          window_size = 5;
          b = (1 / window_size) * ones(1, window_size);
          a = 1;
          filtered_data = zeros(size(rate_temp_z));
          for i = 1 : size(rate_temp_z, 1)
              filtered_data(i, :) = filter(b, a, rate_temp_z(i, :));
          end

          % Add each trial to database
          filtered_database{n, k} = filtered_data;
          hand_pos_database{n, k} = hand_temp_z;
      end
  end

  % Principle Component Analysis (PCA)
  total_time_bins = 0;
  for n = 1 : num_trials
      for k = 1 : num_angles
          total_time_bins = total_time_bins + size(filtered_database{n, k}, 2);
      end
  end

  pca_array = zeros(total_time_bins, num_neurons);
  hand_pos_array = zeros(total_time_bins, num_axis);
  current_row = 1;
  for n = 1 : num_trials
      for k = 1 : num_angles
          current_bin = size(filtered_database{n, k}, 2);
          pca_array(current_row : current_row + current_bin - 1, :) = filtered_database{n, k}';
          hand_pos_array(current_row : current_row + current_bin - 1, :) = hand_pos_database{n, k}';
          current_row = current_row + current_bin;
      end
  end

  [coeff, score, latent, tsquared, explained] = pca(pca_array);

  hand_pos_array_x = hand_pos_array(:, 1);
  hand_pos_array_y = hand_pos_array(:, 2);

  % Training the linear regression model
  X = score(:,:);
  y = hand_pos_array_x(:,:);
  z = hand_pos_array_y(:,:);
    
  % Split data into training and testing sets (e.g., 80% training, 20% testing)
  split_ratio = 0.8;
  split_idx = round(split_ratio * size(X, 1));
  X_train = X(1:split_idx,:);
  X_test = X(split_idx+1:end,:);
  y_train = y(1:split_idx,:);
  y_test = y(split_idx+1:end,:);
  z_train = z(1:split_idx,:);
  z_test = z(split_idx+1:end,:);
    
  % Train linear regression model
  modelx = fitlm(X_train, y_train);
  modely = fitlm(X_train, z_train);
    
  % Evaluate the model's performance
  x_pred = predict(modelx, X_test);
  mse = mean((y_test - x_pred).^2);
  r2 = 1 - mse / var(y_test);

  y_pred = predict(modely, X_test);
  mse1 = mean((z_test - y_pred).^2);
  r2_1 = 1 - mse1 / var(z_test);
    
  % Analyze coefficients (optional)
  coefficients = modelx.Coefficients;
  coefficients1 = modely.Coefficients;
    
  % Display performance metrics
  fprintf('Mean Squared Error (MSE) for x: %.4f\n', mse);
  fprintf('R^2 Score for x: %.4f\n', r2);
  disp(coefficients);

  fprintf('Mean Squared Error (MSE) for y: %.4f\n', mse1);
  fprintf('R^2 Score for y: %.4f\n', r2_1);
  disp(coefficients1);

  modelParameters = {coefficients, coefficients1};

end

