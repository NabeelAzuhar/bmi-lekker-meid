% Last edit: 19/03/22
% Authors: Nabeel Azuhar Mohammed, Gloria Sun, Ioana Lazar, Alexia Badea

function [x,y,modelParameters]= positionEstimator(testData, modelParameters)

    % ---------------------------------------------------------------------
    % Inputs:
    % testData: struct with very similar formatting to trial, other than it has the additinal field of starting position
    % modelParameters: previously saved modelParameters from PCA-LDA analysis using the training dataset
    
    % Outputs:
    % decodedPosX: predicted X position according to the PCR model
    % decodedPosY: predicted Y position according to the PCR model
    % newParameters: any modifications in classification etc stored here
    %----------------------------------------------------------------------
            
    % Initialisations
    binSize = 20; % binning resolution (ms)
    window = 50; % window length (ms)
    timeStart = 320; % start time of testing (ms)

    % Data pre-processing
    dataProcessed = bin_and_sqrt(testData, binSize, 1); % bin and squareroot data
    dataFinal = get_firing_rates(dataProcessed, binSize, window); % convert binned data into firing rate (including smoothing)
    timeTotal = size(testData.spikes, 2); % total (ending) time, start time is 320ms, algorithm iterated every 20ms
    
    % Determine label
    if timeTotal <= 560
    
        dataFinal.rates(modelParameters.lowFirers{1}, :) = []; % drop neuron data with low firing rates
        firingData = reshape(dataFinal.rates, [], 1); % reshape firing rate data into one column
        binCount = (timeTotal/binSize) - (timeStart/binSize) + 1; % actual count of bins

        % get weights from the model parameters outputted from training
        optimTrain = modelParameters.classify(binCount).wOpt_kNN;
        meanFiringTrain = modelParameters.classify(binCount).mFire_kNN;
        WTest = optimTrain' * (firingData - meanFiringTrain); % weights for testing
        WTrain = modelParameters.classify(binCount).wLDA_kNN; % weights for training

        % compute label
        label = get_knns(WTest, WTrain);
        modelParameters.actualLabel = label;
        if label ~= modelParameters.actualLabel
            label = modelParameters.actualLabel;
        end
    
    else % i.e. just keep using the parameters derived with the largest length of training time
        dataFinal.rates(modelParameters.lowFirers{1}, :) = []; % drop neuron data with low firing rates
        firingData = reshape(dataFinal.rates, [], 1); % reshape firing rate data into one column
        label = modelParameters.actualLabel;
    end
    
    % Use outputted label to predict x and y positions
    if timeTotal <= 560
        
        % Compute x and y
        binCount =  (timeTotal/binSize) - (timeStart/binSize) + 1;
        xAverage = modelParameters.averages(binCount).avX(:, label);
        yAverage = modelParameters.averages(binCount).avY(:, label);
        meanFiring = modelParameters.pcr(label, binCount).fMean;
        xCoeff = modelParameters.pcr(label, binCount).bx;
        yCoeff = modelParameters.pcr(label, binCount).by;
        x = (firingData - mean(meanFiring))'*xCoeff + xAverage;
        y = (firingData - mean(meanFiring))'*yCoeff + yAverage;
                
                WTrain = modelParameters.classify(indexer).wLDA_kNN;
                pcaDim = modelParameters.classify(indexer).dPCA_kNN;
                ldaDim = modelParameters.classify(indexer).dLDA_kNN;
                optimTrain = modelParameters.classify(indexer).wOpt_kNN;
                meanFiringTrain = modelParameters.classify(indexer).mFire_kNN;
                % not sure whether it should be the mean from train or test
                WTest = optimTrain'*(firingData-meanFiringTrain); 
                
                
                outLabel = getKnns(WTest, WTrain);
                modelParameters.actualLabel = outLabel;
                if outLabel ~= modelParameters.actualLabel
                    outLabel = modelParameters.actualLabel;
                    
                end

        try
            x = x(timeTotal, 1);
            y = y(timeTotal, 1);
        catch
            x =  x(end, 1);
            y = y(end, 1);
        end
    
    elseif timeTotal > 560 % i.e. just keep using the model with the largest length of training time
        
        xAverage = modelParameters.averages(13).avX(:,label);
        yAverage = modelParameters.averages(13).avY(:,label);
        xCoeff = modelParameters.pcr(label,13).bx;
        yCoeff = modelParameters.pcr(label,13).by;
        
        x = (firingData(1:length(xCoeff)) - mean(firingData(1:length(xCoeff))))'*xCoeff + xAverage;
        y = (firingData(1:length(yCoeff)) - mean(firingData(1:length(yCoeff))))'*yCoeff + yAverage;
        
        try
            x =  x(timeTotal,1);
            y = y(timeTotal,1);
        catch
            x =  x(end,1);
            y = y(end,1);
        end
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function trialProcessed = bin_and_sqrt(trial, group, to_sqrt)
% 
% % Use to re-bin to different resolutions and to sqrt binned spikes (is used
% % to reduce the effects of any significantly higher-firing neurons, which
% % could bias dimensionality reduction)
% 
% % trial = the given struct
% % group = new binning resolution - note the current resolution is 1ms
% % to_sqrt = binary , 1 -> sqrt spikes, 0 -> leave
% 
%     trialProcessed = struct;
%     
% 
%     for i = 1: size(trial,2)
%         for j = 1: size(trial,1)
% 
%             all_spikes = trial(j,i).spikes; % spikes is no neurons x no time points
%             no_neurons = size(all_spikes,1);
%             no_points = size(all_spikes,2);
%             t_new = 1: group : no_points +1; % because it might not round add a 1 
%             spikes = zeros(no_neurons,numel(t_new)-1);
% 
%             for k = 1 : numel(t_new) - 1 % get rid of the paddded bin
%                 spikes(:,k) = sum(all_spikes(:,t_new(k):t_new(k+1)-1),2);
%             end
% 
%             if to_sqrt
%                 spikes = sqrt(spikes);
%             end
% 
%             trialProcessed(j,i).spikes = spikes;
% %             trialProcessed(j,i).handPos = trial(j,i).handPos(1:2,:);
% %             trialProcessed(j,i).bin_size = group; % recorded in ms
%         end
%     end
%     
% end
% 
% 
% function trialFinal = get_firing_rates(trialProcessed,group,scale_window)
% 
% % trial = struct , preferably the struct which has been appropaitely binned
% % and had low-firing neurons removed if needed
% % group = binning resolution - depends on whether you have changed it with
% % the bin_and_sqrt function
% % scale_window = a scaling parameter for the Gaussian kernel - am
% % setting at 50 now but feel free to mess around with it
% 
%     trialFinal = struct;
%     win = 10*(scale_window/group);
%     normstd = scale_window/group;
%     alpha = (win-1)/(2*normstd);
%     temp1 = -(win-1)/2 : (win-1)/2;
%     gausstemp = exp((-1/2) * (alpha * temp1/((win-1)/2)) .^ 2)';
%     gaussian_window = gausstemp/sum(gausstemp);
%     
%     for i = 1: size(trialProcessed,2)
% 
%         for j = 1:size(trialProcessed,1)
%             
%             hold_rates = zeros(size(trialProcessed(j,i).spikes,1),size(trialProcessed(j,i).spikes,2));
%             
%             for k = 1: size(trialProcessed(j,i).spikes,1)
%                 
%                 hold_rates(k,:) = conv(trialProcessed(j,i).spikes(k,:),gaussian_window,'same')/(group/1000);
%             end
%             
%             trialFinal(j,i).rates = hold_rates;
% %             trialFinal(j,i).handPos = trialProcessed(j,i).handPos;
% %             trialFinal(j,i).bin_size = trialProcessed(j,i).bin_size; % recorded in ms
%         end
%     end
% 
% end


    function [labels] = getKnns(testingData, trainingData)
    %GetKnns Predicts labels using k-nearest neighbors algorithm.
    %   
    %   Inputs:
    %       testingData: DimLda x no. test trials, corresponding to the
    %                    projection of the trial data after use of PCA-LDA
    %       trainingData: DimLda x no. training trials, corresponding to the
    %                     projection of the trial data after use of PCA-LDA
    %
    %   Outputs:
    %       labels: Reaching angle/direction labels of the testing data deduced 
    %               with the k-nearest neighbors algorithm

    % Reformatting the train and test data
    trainMatrix = trainingData';
    testMatrix = testingData;
    trainSquareSum = sum(trainMatrix .* trainMatrix, 2);
    testSquareSum = sum(testMatrix .* testMatrix, 1);

    % Calculate distances
    allDists = trainSquareSum(:, ones(1, length(testMatrix))) ...
                + testSquareSum(ones(1, length(trainMatrix)), :) ...
                - 2 * trainMatrix * testMatrix;
    allDists = allDists';

    % Sort for the k nearest neighbors
    k = 25; % Or you can calculate it based on the length of the training data
    [~, sorted] = sort(allDists, 2);
    nearest = sorted(:, 1:k);

    % Determine mode direction for these k-nearest neighbors
    noTrain = size(trainingData, 2) / 8;
    dirLabels = [ones(1, noTrain), 2 * ones(1, noTrain), ...
                  3 * ones(1, noTrain), 4 * ones(1, noTrain), ...
                  5 * ones(1, noTrain), 6 * ones(1, noTrain), ...
                  7 * ones(1, noTrain), 8 * ones(1, noTrain)]';
    nearestLabels = reshape(dirLabels(nearest), [], k);
    labels = mode(mode(nearestLabels, 2));

end




end