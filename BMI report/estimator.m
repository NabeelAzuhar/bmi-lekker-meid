function [x, y, modelParameters]= estimator(testData, modelParameters)

    % ---------------------------------------------------------------------
    % Inputs:
        % testData: 1 trial with the additinal field of starting position
        % modelParameters: previously saved modelParameters from PCA-LDA analysis using the training dataset
    
    % Outputs:
        % decodedPosX: predicted X position
        % decodedPosY: predicted Y position
        % modelParameters: updated parameters
    %----------------------------------------------------------------------
    
    % Initialisations
        binSize = modelParameters.binSize; % binning resolution (ms)
        window = modelParameters.window; % window length (ms)
        startTime = 320; % start time of testing (ms)
        endTime = modelParameters.endTime; % minimum time length in training data
%         % labels for selecting data for a selected angle from firing Data
%         y = repmat(1:8, 100, 1);
%         y = y(:);
%         disp(size(y)) % (800 x 1)
    
    % 1. Data Pre-processing
        dataProcessed = preprocess(testData, binSize, window); % binned, squarerooted, smoothed - 1 single struct
        dataProcessed.rates(modelParameters.lowFirers, :) = []; % drop neuron data with low firing rates
        timeTotal = size(testData.spikes, 2); % total (ending) time of the trial given in ms
        binCount = (timeTotal/binSize) - (startTime/binSize) + 1; % bin indices to indicate which classification parameters to use
        
        % Reformat data
        firingData = reshape(dataProcessed.rates, [], 1)'; % reshape firing rate data into one column
        % Out: firingData = (1 x 2660)

    %%% Classification %%%
        if timeTotal <= endTime % if total time is within the preset training time bins
            
            % Knn
            [label] = calcKnn(firingData, modelParameters, binCount);
            modelParameters.actualLabel = label;
        
        else % if time goes beyond what's been trained, just keep using the parameters derived with the largest length of training time
            label = modelParameters.actualLabel;
         
        end % end of KNN classification
    %%%%     
    
    % 3. Use outputted label to predict x and y positions using Principal Component Regression
        if timeTotal <= endTime
            
        % 3.1. Compute x and y using the predicted label
            [x, y] = pcr(firingData, modelParameters, binCount, label, 0);
    
            % get end position of interval
            try
                x = x(timeTotal, 1);
                y = y(timeTotal, 1);
            catch
                x =  x(end, 1);
                y = y(end, 1);
            end
        
        else % i.e. just keep using the model with the largest length of training time bin
            
            [x, y] = pcr(firingData, modelParameters, binCount, label, 1);
    
            % get end position of interval
            try
                x = x(timeTotal, 1); % x is PADDED!
                y = y(timeTotal, 1);
            catch
                x = x(end, 1);
                y = y(end, 1);
            end
    
        end % end of x and y position decoding




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function [processedData] = preprocess(data, binSize, window)
        % Initialisations
        processedData = struct; % output
        numNeurons = size(data(1,1).spikes, 1);
            
        % Binning & Squarerooting - 20ms bins, sqrt to avoid large values
        for angle = 1 : size(data, 2)
            for trial = 1 : size(data, 1)
         
                % Initialisations
                spikeData = data(trial, angle).spikes; % extract spike data (98 x time steps)
                totalTime = size(spikeData, 2); % total number of time steps in ms
                binIndices = 1 : binSize : totalTime+1; % start of each time bin in ms
                spikeBins = zeros(numNeurons, length(binIndices)-1); % initialised binned spike data, (98 x number of bins)
    
                % Bin then squareroot the spike data            
                for bin = 1 : length(binIndices) - 1 % iterate through each bin
                    spikeBins(:, bin) = sum(spikeData(:, binIndices(bin):binIndices(bin+1)-1), 2); % sum spike number
                end
                spikeBins = sqrt(spikeBins);
    
                % fill up the output
                processedData(trial, angle).spikes = spikeBins; % spikes are now binned
            end
        end
    
        % Convert spike count per bin into firing rate + Gaussian window for smoothing
        % Generating the Gaussian window
        windowWidth = 10 * (window/binSize); % width of gaussian window
        std = window/binSize; % normalised std
        alpha = (windowWidth-1) / (2*std); % determines spread of curve (exp coefficient)
        tmp = -(windowWidth-1)/2 : (windowWidth-1)/2; % symmetric vector ranging about 0
        gaussTemp = exp((-1/2) * (alpha*tmp/((windowWidth-1)/2)).^2)'; % gaussian window
        gaussWindow = gaussTemp/sum(gaussTemp); % normalised gaussian window, FINAL to use
    
        % Add smoothened firing rates to the processed data
        for angle = 1 : size(processedData, 2)
            for trial = 1 : size(processedData, 1)
                % rates field to be added
                firingRates = zeros(size(processedData(trial, angle).spikes, 1), size(processedData(trial, angle).spikes, 2)); 
                % convolve window with each neuron for smoothing
                for neuron = 1 : size(processedData(trial, angle).spikes, 1)
                    firingRates(neuron, :) = conv(processedData(trial, angle).spikes(neuron, :), gaussWindow, 'same') / (binSize/1000);
                end
                processedData(trial, angle).rates = firingRates; % add rates as a new field to processed data
            end
        end
    end


    function [label] = calcKnn(firingData, modelParameters, binCount)
        % Get classification weights from the model parameters for KNN for the corresponding bin interval
        transformation = modelParameters.knnClassify(binCount).transformation; % (2660 x 6)
        firingMean = modelParameters.knnClassify(binCount).meanFiring; % mean firing rate % (1 x 2660)
        testLDA = (firingData-firingMean) * transformation; % (1 x 6), test data projected onto LDA components
        trainLDA = modelParameters.knnClassify(binCount).trainProjected; % (800 x 6), train data projected onto LDA components
        numTrain = size(trainLDA, 1) / 8;
        dirLabels = [ones(1, numTrain), 2 * ones(1, numTrain), ...
                      3 * ones(1, numTrain), 4 * ones(1, numTrain), ...
                      5 * ones(1, numTrain), 6 * ones(1, numTrain), ...
                      7 * ones(1, numTrain), 8 * ones(1, numTrain)]';

        % KNN
        num_neighbors = 25;  % Choose the number of neighbors for KNN
        label = predict(fitcknn(trainLDA, dirLabels, 'NumNeighbors', num_neighbors), testLDA);
         
    end

    
    function [x, y] = pcr(X, modelParameters, binCount, label, mode)
        if mode == 0
            % Principal Component Analysis for dimensionality reduction
                X_pca = X * modelParameters.regression(label, binCount).pcaTransformation;
    
            % Linear regression
                x = predict(modelParameters.regression(label, binCount).mdlx, X_pca) + modelParameters.regression(label, binCount).meanx;
                y = predict(modelParameters.regression(label, binCount).mdly, X_pca) + modelParameters.regression(label, binCount).meany;
        else
            % Principal Component Analysis for dimensionality reduction
                X_pca = X(:, size(modelParameters.regression(label, end).pcaTransformation, 2)) * modelParameters.regression(label, end).pcaTransformation;
    
            % Linear regression
                x = predict(modelParameters.regression(label, end).mdlx, X_pca) + modelParameters.regression(label, end).meanx;
                y = predict(modelParameters.regression(label, end).mdly, X_pca) + modelParameters.regression(label, end).meany;
        end
            
    end
end