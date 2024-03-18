function  [trial_,concatenated_pos] = trainingtest(trainingData)
[nTrials, nDirections] = size(trainingData);
binSize = 20;
std = 50;

% reachAngles = [30 70 110 150 190 230 310 350]; % given in degrees

modelParameters = struct;
startTime = 320;
endTime = 560; %??



% preprocessing
 [trial_,minTrialLen]= prep(trainingData,binSize,std);
% concatenate trials
% add get min trial length
concatenated_array =[];
concatenated_pos =[];
for dir = 1:nDirections
    for trialN = 1:nTrials
        concatenated_array = [concatenated_array; trial_(trialN, dir).rates(:,1:minTrialLen)'];
        concatenated_pos = [concatenated_pos; trial_(trialN, dir).handPos(:,1:minTrialLen)'];
    end
end


disp(size(concatenated_array))
% remove low firing neurons 
removers = [];
firingThreshold = 0.5; %Hz
k = 1;
disp(size(concatenated_array,2))
for neuron=1:size(concatenated_array,2)
   mean_firing_rate = mean(concatenated_array(:,neuron));
   disp(mean_firing_rate)
   if mean_firing_rate < firingThreshold
       removers(k) = neuron; 
       k=k+1;
   end 
end
% concatenated_array(:,removers)=[];
% modelParameters.lowFirers = removers;
disp(size(concatenated_array))

% PCA

[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(concatenated_array);

feature_matrix = SCORE(:,1:10);
disp(size(feature_matrix))

% add history bins
n_history_bins = 3;
n_features = size(feature_matrix,2)*(n_history_bins+1);
n_timebins = size(feature_matrix,1);
new_feature_matrix = zeros(n_timebins,n_features);
for timebin = n_history_bins+1:size(feature_matrix,1)
    new_feature_matrix(timebin,:) =  reshape(feature_matrix(timebin-n_history_bins:timebin,:)', 1, []);
end 

% remove first n_history_bins from each trial


disp(size(new_feature_matrix))
timebins_per_trial = n_timebins/(nTrials*nDirections);
% feature_matrix_ = zeros(n_history_bins*nTrials*nDirections);
rows_to_delete = [];
for n_trial=1:nTrials*nDirections
    rows_to_delete = cat(1, rows_to_delete, (1 + (n_trial - 1) * timebins_per_trial : (n_trial - 1) * timebins_per_trial + n_history_bins)');
end

new_feature_matrix(rows_to_delete, :) = [];
concatenated_pos(rows_to_delete, :) = [];
disp(size(new_feature_matrix))
disp(size(concatenated_pos))





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [trial_,Min] = prep(trainingData,binSize,std)
    [ntrials, ndirections] = size(trainingData);
    Min = Inf;
    trial_ = trainingData;
    for trialNum = 1:ntrials
        for direction = 1:ndirections
            trial_(trialNum, direction) = combineTimeBeans(trial_(trialNum, direction), binSize);
            trial_(trialNum, direction) = sqrtTransformSignal(trial_(trialNum, direction));
        Min = min(Min,size(trial_(trialNum,direction).spikes,2));
        end
    end

    for trialNum = 1:ntrials
        for direction = 1:ndirections
             trial_(trialNum, direction).rates= addFiringRates(trial_(trialNum, direction), std,binSize);
        end
    end
 end
    
function adjustedTrial = combineTimeBeans(trial_ed, bin_size)

    [N, T] = size(trial_ed.spikes);
    T = floor(T/bin_size);
    adjusted_spikes = zeros(N, T);
    adjusted_handPos = zeros(2, T);
    adjustedTrial.trialId = trial_ed.trialId;
    for t = 1:T
        for n = 1:N
        adjusted_spikes(n, t) = sum(trial_ed.spikes(n,bin_size*(t-1)+1:bin_size*t));
        end
    for i =1:2
        adjusted_handPos(i,:) = sum(trial_ed.handPos(i,1:bin_size:end));
    end
    end
    adjustedTrial.spikes = adjusted_spikes;
    adjustedTrial.handPos = adjusted_handPos;
end

function sqrtSignal = sqrtTransformSignal(trial_ed)

    [N, T] = size(trial_ed.spikes);
    sqrt_spikes = zeros(N, T);
    sqrtSignal.handPos = trial_ed.handPos;
    sqrtSignal.trialId = trial_ed.trialId;
    
    for t = 1:T
        sqrt_spikes(:, t) = sqrt(trial_ed.spikes(:,t));
    
    end
    
    sqrtSignal.spikes = sqrt_spikes;


end

function adjusted_rates = addFiringRates(trial_ed, std, bin_size)
   
    win = customGaussWin(floor(10*std/bin_size), std/bin_size);
    
    adjusted_rates = convn(trial_ed.spikes, win, 'same');

end

function win = customGaussWin(N, sigma)
    % N: window length, sigma: standard deviation of the Gaussian window
    
    % Ensure N is an integer
    N = floor(N);
    
    % Generate a linear space from -(N-1)/2 to (N-1)/2
    n = linspace(-(N-1)/2, (N-1)/2, N);
    
    % Calculate the Gaussian window
    win = exp(-0.5 * (n / sigma).^2);
    
    % Normalize the window (optional, depending on your use case)
    win = win / sum(win);
end
end 
