% Last edit: 19/03/22
% Authors: Nabeel Azuhar Mohammed, Gloria Sun, Ioana Lazar, Alexia Badea

function [x, y, modelParameters]= positionEstimatorIoana(testData, modelParameters)

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
    window = 30; % window length (ms)
    timeStart = 320; % start time of testing (ms)
    [numTrials, numDirections] = size(testData);
    numNeurons = size(testData(1, 1).spikes, 1); % stays constant throughout
    binSize = 20; % binning resolution (ms)
    window = 30; % smoothing gaussian window std (ms)
    startTime = 320; % start time of testing (ms)
    timeEnd = 560; % smallest time length in training data rounded to time bin of 20ms
    nHistBins = 10; % number of past feature bins used to predict current position
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
    

% 2. Determine label by KNN classification
    if timeTotal <= endTime % if total time is within the preset training time bins
 
        % get classification weights from the model parameters for KNN for the corresponding bin interval
        testProjection = modelParameters.knnClassify(binCount).testProjection; % (2660 x 6)
        firingMean = modelParameters.knnClassify(binCount).meanFiring; % mean firing rate % (2660 x 1)
%         disp(size(firingData))
%         disp(size(testProjection))
        testLDA = testProjection' * (firingData - firingMean); % (6 x 1), test data projected onto LDA components
        trainLDA = modelParameters.knnClassify(binCount).trainProjected; % (6 x 800), train data projected onto LDA components

        % compute label using KNN
        label = calcKnns(testLDA, trainLDA); % label = predicted direciton using knn
        modelParameters.actualLabel = label;
    
    else % if time goes beyond what's been trained, just keep using the parameters derived with the largest length of training time
        label = modelParameters.actualLabel;
     
    end % end of KNN classification
 
%%%%%%%%%%%%% REGRESSION %%%%%%%%%%%%

% 3. Predict hand position using the PCR coefficients obtained in training
% for each label
   
    firingData =dataProcessed.rates(:,end-nHistBins:end)';

    xCoeff = modelParameters.pcr(label).xM;
    yCoeff = modelParameters.pcr(label).yM;

    % pcaData = (firingData-mean(firingData))*modelParameters.pcaTransform;
    pcaData = (firingData)*modelParameters.pcaTransform;
    pcaData = reshape(pcaData', 1, []);

    x = pcaData * xCoeff + modelParameters.averages(label).xMean + testData.startHandPos(1);
    y = pcaData* yCoeff + modelParameters.averages(label).yMean + testData.startHandPos(2);

    % x = pcaData * xCoeff + modelParameters.averages(label).xMean;
    % y = pcaData* yCoeff + modelParameters.averages(label).yMean;


    try
        x = x(timeTotal, 1);
        y = y(timeTotal, 1);
    catch
        x =  x(end, 1);
        y = y(end, 1);
    end


    
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
    % GetKnns Predicts labels using k-nearest neighbors algorithm.
    %   
    %   Arguments:
    %       testingData: DimLda x no. test trials, corresponding to the
    %                    projection of the trial data after use of PCA-LDA
    %       trainingData: DimLda x no. training trials, corresponding to the
    %                     projection of the trial data after use of PCA-LDA
    %
    %   Returns:
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

    end % end of KNN function

end