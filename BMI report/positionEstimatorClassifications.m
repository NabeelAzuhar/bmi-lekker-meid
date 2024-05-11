% Last edit: 19/03/22
% Authors: Nabeel Azuhar Mohammed, Gloria Sun, Ioana Lazar, Alexia Badea

function [x, y, modelParameters]= positionEstimatorClassifications(testData, modelParameters, trial_split)

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

% 1. Data Pre-processing
    dataProcessed = dataProcessor(testData, binSize, window); % binned, squarerooted, smoothed - 1 single struct
    dataProcessed.rates(modelParameters.lowFirers, :) = []; % drop neuron data with low firing rates
    timeTotal = size(testData.spikes, 2); % total (ending) time of the trial given in ms
    binCount = (timeTotal/binSize) - (startTime/binSize) + 1; % bin indices to indicate which classification parameters to use
    
    % Reformat data
    firingData = reshape(dataProcessed.rates, [], 1); % reshape firing rate data into one column
    % Out: firingData = (2660x1)
    

% 2. Determine label by classification
    if timeTotal <= endTime % if total time is within the preset training time bins
 
        % get classification weights from the model parameters for KNN for the corresponding bin interval
        testProjection = modelParameters.knnClassify(binCount).testProjection; % (2660 x 6)
        firingMean = modelParameters.knnClassify(binCount).meanFiring; % mean firing rate % (2660 x 1)
        testLDA = testProjection' * (firingData - firingMean); % (6 x 1), test data projected onto LDA components
        trainLDA = modelParameters.knnClassify(binCount).trainProjected; % (6 x 800), train data projected onto LDA components
        ecoc_model = modelParameters.knnClassify(binCount).ecoc_model;

%         % compute label using KNN
%         label = calcKnns(testLDA, trainLDA); % label = predicted direciton using knn
%         modelParameters.actualLabel = label;

        % compute label using Linear SVM
        label = calcLinearSVM(testLDA, ecoc_model); % label = predicted direciton using linear svm
        modelParameters.actualLabel = label;

    
    else % if time goes beyond what's been trained, just keep using the parameters derived with the largest length of training time
        label = modelParameters.actualLabel;
        modelParameters.actualLabel = label;
     
    end % end of classification
 

% 3. Use outputted label to predict x and y positions using Principal Component Regression
    if timeTotal <= endTime
        
    % 3.1. Compute x and y using the predicted label
        xMean = modelParameters.positionMeans(binCount).xMean(:, label);
        yMean = modelParameters.positionMeans(binCount).yMean(:, label);
        firingMean = modelParameters.regression(label, binCount).firingMean;
        xCoeff = modelParameters.regression(label, binCount).xCoeff;
        yCoeff = modelParameters.regression(label, binCount).yCoeff;
        x = ((firingData - mean(firingMean)))'* xCoeff + xMean; % (792, 1) - decoded x for 792 time bins
        y = ((firingData - mean(firingMean)))'* yCoeff + yMean;

        % get end position of interval
        try
            x = x(timeTotal, 1);
            y = y(timeTotal, 1);
        catch
            x =  x(end, 1);
            y = y(end, 1);
        end
    
    else % i.e. just keep using the model with the largest length of training time bin
        
        xMean = modelParameters.positionMeans(end).xMean(:, label);
        yMean = modelParameters.positionMeans(end).yMean(:, label);
        xCoeff = modelParameters.regression(label, end).xCoeff;
        yCoeff = modelParameters.regression(label, end).yCoeff;
        x = (firingData(1:length(xCoeff)) - mean(firingData(1:length(xCoeff))))' * xCoeff + xMean;
        y = (firingData(1:length(yCoeff)) - mean(firingData(1:length(yCoeff))))' * yCoeff + yMean;

        % get end position of interval
        try
            x = x(timeTotal, 1); % x is PADDED!
            y = y(timeTotal, 1);
        catch
            x = x(end, 1);
            y = y(end, 1);
        end

    end % end of x and y position decoding



% Nested functions --------------------------------------------------------

    function [dataProcessed] = dataProcessor(data, binSize, window)
    %----------------------------------------------------------------------
        % Re-bins data and squareroot to reduce the effects of anamolies
        % Transforms spike data into firing rate data

        % Arguments:
        %   data: spike data to be processed
        %   binSize: binning resolution (time window per bin)
        %   window: window length for smoothing

        % Return Value:
        %   dataProcessed: binned, smoothed firing rate data
    %----------------------------------------------------------------------
        
    % Initialisations
        dataProcessed = struct; % output
        numNeurons = size(data(1,1).spikes, 1);
        
    % Binning & Squarerooting - 20ms bins, sqrt to avoid large values
        for angle = 1 : size(data, 2)
            for trial = 1 : size(data, 1)
         
                % initialisations
                spikeData = data(trial, angle).spikes; % extract spike data (98 x time steps)
                totalTime = size(spikeData, 2); % total number of time steps
                binStarts = 1 : binSize : totalTime+1; % starting time stamps of each bin
                spikeBins = zeros(numNeurons, numel(binStarts)-1); % binned data, (98 x number of bins)
                
                % bin then squareroot the data            
                for bin = 1 : numel(binStarts)-1 % iterate through each bin
                    spikeBins(:, bin) = sum(spikeData(:, binStarts(bin):binStarts(bin+1)-1), 2); % sum spike number
                end
                spikeBins = sqrt(spikeBins);

                % fill up the output
                dataProcessed(trial, angle).spikes = spikeBins; % spikes are now binned
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

        % add smoothened firing rates to the processed data
        for angle = 1 : size(dataProcessed, 2)
            for trial = 1 : size(dataProcessed, 1)
                % rates field to be added
                firingRates = zeros(size(dataProcessed(trial, angle).spikes, 1), size(dataProcessed(trial, angle).spikes, 2));
                
                % convolve window with each neuron for smoothing
                for neuron = 1 : size(dataProcessed(trial, angle).spikes, 1)
                    firingRates(neuron, :) = conv(dataProcessed(trial, angle).spikes(neuron, :), gaussWindow, 'same') / (binSize/1000);
                end
                dataProcessed(trial, angle).rates = firingRates; % add rates as a new field to processed data
            end
        end

    end % end of function dataProcessor


    function [labels] = calcKnns(testingData, trainingData)
    %----------------------------------------------------------------------
    % calcKnns Predicts labels using k-nearest neighbors algorithm.
    %   
    %   Arguments:
    %       testingData: DimLda x no. test trials (1), corresponding to the projection of the trial data onto LDA components               
    %       trainingData: DimLda x no. training trials, corresponding to the projection of the trial data onto LDA components  
    %
    %   Returns:
    %       labels: Reaching angle/direction labels of the testing data deduced with the k-nearest neighbors algorithm      
    %----------------------------------------------------------------------

    % Reformatting the train and test data
    trainMat = trainingData'; % training data projected onto LDA
    testMat = testingData; % testing data projected onto LDA
    trainSquaredSum = sum(trainMat .* trainMat, 2);
    testSquaredSum = sum(testMat .* testMat, 1);

    % Calculate distances
    distances = trainSquaredSum(:, ones(1, length(testMat))) ...
                + testSquaredSum(ones(1, length(trainMat)), :) ...
                - 2 * trainMat * testMat; % calculates the Euclidean distance between each pair of training and testing data points

    % Sort for the k nearest neighbors
    k = 25; % Or you can calculate it based on the length of the training data
    [~, sorted] = sort(distances', 2);
    nearest = sorted(:, 1: k);

    % Determine the direction for the k-nearest neighbors
    numTrain = size(trainingData, 2) / 8;
    dirLabels = [ones(1, numTrain), 2 * ones(1, numTrain), ...
                  3 * ones(1, numTrain), 4 * ones(1, numTrain), ...
                  5 * ones(1, numTrain), 6 * ones(1, numTrain), ...
                  7 * ones(1, numTrain), 8 * ones(1, numTrain)]';
    nearestLabels = reshape(dirLabels(nearest), [], k);
    labels = mode(mode(nearestLabels, 2));

    end % end of KNN function


    function [labels] = calcLinearSVM(testingData, ecoc_model)
    %----------------------------------------------------------------------
    % calcKnns Predicts labels using k-nearest neighbors algorithm.
    %   
    %   Arguments:
    %       testingData: DimLda x no. test trials (1), corresponding to the projection of the trial data onto LDA components               
    %       ecoc_model: Pretrained Linear SVM model  
    %
    %   Returns:
    %       labels: Reaching angle/direction labels of the testing data deduced with linear SVM      
    %----------------------------------------------------------------------
    testMat = testingData'; % testing data projected onto LDA

    % Step 3: Make predictions on the test data
    labels = predict(ecoc_model, testMat);
    end

% Nested functions --------------------------------------------------------

end % end of positionEstimator function