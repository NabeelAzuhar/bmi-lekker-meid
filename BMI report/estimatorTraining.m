function [modelParameters] = estimatorTraining(trainingData)

    % initialisations
    modelParameters = struct; % creates modelParameters structure
    numNeurons = size(trainingData(1, 1).spikes, 1); % stays constant throughout
    numTrials = size(trainingData, 1);
    numDirections = size(trainingData, 2);

    %%% Preprocessing Data %%%
    [processedData, binSize, window] = preprocess(trainingData, numNeurons);
    modelParameters.binSize = binSize; % adds binSize to modelParameters
    modelParameters.window = window; % adds window to modelParameters

    % determine max and min number of time steps
    timeSteps = [];
    for angle = 1 : numDirections
        for trial = 1 : numTrials
            timeSteps = [timeSteps, size(trainingData(trial, angle).handPos, 2)];
        end
    end
    startTime = 320;
    maxTimeSteps = max(timeSteps);
    endTime = floor(min(timeSteps)/binSize) * binSize; % smallest time length in training data rounded to time bin of 20ms
    modelParameters.endTime = endTime; % adds endTime to modelParameters

    % labels for selecting data for a selected angle from firing Data
    labels = repmat(1:numDirections, numTrials, 1);
    labels = labels(:);

    %%% Restructure training data into analysis appropriate format %%%
    % Initialisations
    firingData = zeros(numNeurons*endTime/binSize, numDirections*numTrials); % firing data (binned, truncated to 28 bins)
    handPosData = zeros(2*endTime/binSize, numTrials*numDirections); % handPos data (binned, truncated to 28 bins)
    xPadded = zeros(numTrials, maxTimeSteps, numDirections); % original handPos data (in ms) but padded with the longest trajectory
    yPadded = zeros(numTrials, maxTimeSteps, numDirections);
    
    % Generate firing and hand position data matrices
    for angle = 1 : numDirections % each angle
        for trial = 1 : numTrials % each trial
            % generate padded hand position data
            xPadded(trial, :, angle) = [trainingData(trial,angle).handPos(1,:), trainingData(trial,angle).handPos(1, end) * ones(1, maxTimeSteps - timeSteps(numTrials*(angle-1) + trial))];
            yPadded(trial, :, angle) = [trainingData(trial,angle).handPos(2,:), trainingData(trial,angle).handPos(2, end) * ones(1, maxTimeSteps - timeSteps(numTrials*(angle-1) + trial))]; 

            for bin = 1 : endTime/binSize % each time bin -> taking 28 time bins based on 560ms total time
                % generate firing data
                firingData(numNeurons*(bin-1)+1 : numNeurons*bin, numTrials*(angle-1)+trial) = processedData(trial, angle).rates(:, bin);   
                % generate handPos data
                handPosData(2*(bin-1)+1 : 2*bin, numTrials*(angle-1)+trial) = processedData(trial, angle).handPos(:, bin);  
            end
        end
    end
    firingData = firingData';
    handPosData = handPosData';

    %%% Remove low firing neurons %%%
    [firingData, numNeurons, lowFiringNeurons] = removeNeurons(firingData, numNeurons);
    modelParameters.lowFirers = lowFiringNeurons; % record low firing neurons to remove the same ones in test data 

    % Classification and Regression
    binIndices = (startTime:binSize:endTime) / binSize; % 16 : 28
    intervalIdx = 1;

    % start with data from 320ms (16th bin), then iteratively add 20ms (1 bin) until 560ms (28th bin) for training
    for interval = binIndices % iteratively add testing time: 16, 17, 18, ... 28
        
        % get firing rate data up to the certain time bin
        firingCurrent = firingData(:, 1:numNeurons*interval);
        overallMean = mean(firingCurrent, 1); % (1 x 2660), mean rate for each neuron-bin
%         disp(size(overallMean))

        %%% Classification %%%
        [trainProjected, transformation, dimPCA, dimLDA] = pcaLda(firingCurrent, labels);

        % Add to ModelParameters
        modelParameters.knnClassify(intervalIdx).trainProjected = trainProjected;
        modelParameters.knnClassify(intervalIdx).transformation = transformation;
        modelParameters.knnClassify(intervalIdx).dimPCA = dimPCA;
        modelParameters.knnClassify(intervalIdx).dimLDA = dimLDA;
        modelParameters.knnClassify(intervalIdx).meanFiring = overallMean; % (1 x 2660), mean rate for each neuron-bin
        %%%

        %%% Regression %%%
        for angle = 1 : numDirections
            % select data for the specified angle
            xPos = handPosData(labels==angle, (startTime/binSize * 2 - 1):2:end); 
            yPos = handPosData(labels==angle, (startTime/binSize * 2):2:end); % (100 x 13), 13 bins from 320ms to 560ms (1, 2, 3... 13)

            % select the firing data that corresponds to the current interval AND the given angle
            firingWindowed = firingCurrent(labels == angle, :); % firing data for current interval, selected angle e.g. (100x2660)

            % Regression
            [mdlx, mdly, meanx, meany, pcaTransformation] = pcrLinear(firingWindowed, xPos, yPos, intervalIdx);
            
            % record model parameters
            modelParameters.regression(angle, intervalIdx).mdlx = mdlx;
            modelParameters.regression(angle, intervalIdx).mdly = mdly;
            modelParameters.regression(angle, intervalIdx).meanx = meanx;
            modelParameters.regression(angle, intervalIdx).meany = meany;
            modelParameters.regression(angle, intervalIdx).pcaTransformation = pcaTransformation; % (2660 x 49)
%             modelParameters.regression(angle, intervalIdx).firingMean = mean(firingWindowed, 1); % (1 x 2660)
%             modelParameters.positionMeans(intervalIdx).xMean = squeeze(mean(xPadded, 1)); % squeeze(mean(xPadded, 1)) = (975 x 8), mean across 100 trials for each angle
%             modelParameters.positionMeans(intervalIdx).yMean = squeeze(mean(yPadded, 1));   % (975 x 8)
        end
        %%%

        intervalIdx = intervalIdx + 1; % record the current bin index (13th bin is the 1st)
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function [processedData, binSize, window] = preprocess(data, numNeurons)
        % Initialisations
        processedData = struct; % output
        binSize = 20;
        window = 30;
            
        % Binning & Squarerooting - 20ms bins, sqrt to avoid large values
        for angle = 1 : size(data, 2)
            for trial = 1 : size(data, 1)
         
                % Initialisations
                spikeData = data(trial, angle).spikes; % extract spike data (98 x time steps)
                handPosData = data(trial, angle).handPos; % extract handPos data
                totalTime = size(spikeData, 2); % total number of time steps in ms
                binIndices = 1 : binSize : totalTime+1; % start of each time bin in ms
                spikeBins = zeros(numNeurons, length(binIndices)-1); % initialised binned spike data, (98 x number of bins)
                handPosBins = zeros(2, length(binIndices)-1); % initialised handPos data (2 directions x number of bins)
    
                % Bin then squareroot the spike data            
                for bin = 1 : length(binIndices) - 1 % iterate through each bin
                    spikeBins(:, bin) = sum(spikeData(:, binIndices(bin):binIndices(bin+1)-1), 2); % sum spike number
                    handPosBins(:, bin) = mean(handPosData(1:2, binIndices(bin):binIndices(bin+1)-1), 2); % sample the hand position at the beginning of each bin
                end
                spikeBins = sqrt(spikeBins);
    
                % fill up the output
                processedData(trial, angle).spikes = spikeBins; % spikes are now binned
                processedData(trial, angle).handPos = handPosBins; % select only x and y
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
    
    
    function [firingData, numNeurons, lowFiringNeurons] = removeNeurons(firingData, numNeurons)
        % Initialisations
        lowFiringNeurons = []; % list to store the indices of low-firing neurons
        removedCols = []; % rows of data to remove (for low firing neurons at all its time bins)
    
        % Identify neurons with low firing rate
        for neuron = 1 : numNeurons
            avgRate = mean(mean(firingData(:, neuron:98:end)));
            if avgRate < 0.5 % remove neurons with average rate less than 0.5
                lowFiringNeurons = [lowFiringNeurons, neuron];
            end
        end
    
        % Remove the neurons
        for lowFirer = lowFiringNeurons
            removedCols = [removedCols, lowFirer:numNeurons:length(firingData)];
        end
        firingData(:, removedCols) = []; % remove the rows for the low firers
    
        % Update numNeurons
        numNeurons = numNeurons - length(lowFiringNeurons); % update number of neurons after removing the low firers
    end
    
    
    function [trainProjected, transformation, dimPCA, dimLDA] = pcaLda(X, labels)
        % Principal Component Analysis for dimensionality reduction
            [coeff, score, ~, ~, explained] = pca(X);
            cumulative_explained = cumsum(explained);
            pcaThreshold = 70;
            dimPCA = find(cumulative_explained >= pcaThreshold, 1);
            X_pca = X * coeff(:, 1:dimPCA);
    
        % Linear Discriminent Analysis (LDA)
            dimLDA = 6;
            lda_model = fitcdiscr(X_pca, labels, 'DiscrimType', 'linear');
            % Get the transformation matrix from LDA model
            lda_vectors = lda_model.Mu;  % Get the means of each class
            lda_vectors = lda_vectors - mean(X_pca);  % Center the LDA vectors
            % Normalize LDA vectors (optional but recommended)
            lda_vectors = lda_vectors ./ vecnorm(lda_vectors);
            % Project X_pca onto the LDA vectors
            X_lda = X_pca * lda_vectors';
            % Select the top 6 components
            [~, idx] = maxk(var(X_lda), dimLDA);  % Select indices of top 6 components
            trainProjected = X_lda(:, idx); % (800 x 6)
    
            % Transformation from X to trainProjected
            transformation = pinv(X) * trainProjected; % (2660 x 6)
    end
    
    
    function [mdlx, mdly, meanx, meany, pcaTransformation] = pcrLinear(X, xPos, yPos, intervalIdx)
        % select data for the current time bin (e.g. 13, 14 ... 28)
            xCurrent = xPos(:, intervalIdx) - mean(xPos(:, intervalIdx)); % (100x1), mean removed
            yCurrent = yPos(:, intervalIdx) - mean(yPos(:, intervalIdx)); % (100x1), mean removed
    
        % Principal Component Analysis for dimensionality reduction
            [coeff, score, ~, ~, explained] = pca(X);
            cumulative_explained = cumsum(explained);
            pcaThreshold = 70;
            dimPCA = find(cumulative_explained >= pcaThreshold, 1);
            X_pca = X * coeff(:, 1:dimPCA);
            pcaTransformation = coeff(:, 1:dimPCA);
        
        % Linear regression
            mdlx = fitlm(X_pca, xCurrent);
            mdly = fitlm(X_pca, yCurrent);
            meanx = mean(xPos(:, intervalIdx));
            meany = mean(yPos(:, intervalIdx));
    end
end
