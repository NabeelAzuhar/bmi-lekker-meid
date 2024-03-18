function [modelParameters] = gloriaTraining(trainingData)

%--------------------------------------------------------------------------
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
%--------------------------------------------------------------------------

% Initialisations
    modelParameters = struct; % output
    numNeurons = size(trainingData(1, 1).spikes, 1); % stays constant throughout
    numTrials = size(trainingData, 1);
    numDirections = size(trainingData, 2);
    binSize = 20; % manually set bin size for data binning, 28 bins
    window = 30; % manually set window length for smoothing
    modelParameters.binSize = binSize;
    modelParameters.window = window;
    
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
    modelParameters.endTime = endTime;

    % labels for selecting data for a selected angle from firing Data
    labels = repmat(1:numDirections, numTrials, 1);
    labels = labels(:);

   
% 1. Data Pre-processing
    dataProcessed = dataProcessor(trainingData, binSize, window); % dataProcessed.rates = firing rates
    % Out: dataProcessed: binned (20ms) & smoothed spikes data, with .rates attribute & binned x, y handPos as .handPos


% 2. Generate firing data matrix + filter low firing neurons
    % 2.1 Fetch firing rate data and create matrix
    firingData = zeros(numNeurons*endTime/binSize, numDirections*numTrials);
    for angle = 1 : numDirections % each angle
        for trial = 1 : numTrials % each trial
            for bin = 1 : endTime/binSize % each time bin -> taking 28 time bins based on 560ms total time
                firingData(numNeurons*(bin-1)+1 : numNeurons*bin, numTrials*(angle-1)+trial) = dataProcessed(trial, angle).rates(:, bin);     
            end
        end
    end
    % Out: firingData (98*28, 8*100): rows = 98 neurons * 28 bins, columns = 8 angles * 100 trials

    % 2.2 Mark the neurons to be removed (with rates < 0.5)
    lowFiringNeurons = []; % list to store the indices of low-firing neurons
    for neuron = 1 : numNeurons
        avgRate = mean(mean(firingData(neuron:98:end, :)));
        if avgRate < 0.5 % remove neurons with average rate less than 0.5
            lowFiringNeurons = [lowFiringNeurons, neuron];
        end
    end
    % Out: lowFiringNeurons: a list containing all the low-firing neurons

    % 2.3 Remove the neurons
    removedRows = []; % rows of data to remove (for low firing neurons at all its time bins)
    for lowFirer = lowFiringNeurons
        removedRows = [removedRows, lowFirer:numNeurons:length(firingData)];
    end
    firingData(removedRows, :) = []; % remove the rows for the low firers
    numNeuronsNew = numNeurons - length(lowFiringNeurons); % new number of neurons after removing the low firers
    modelParameters.lowFirers = lowFiringNeurons; % record low firing neurons to model parameters output
    % Out:
    %   numNeuronsNew: updated number of neurons after rate filtering
    %   firingData: filtered rows after neuron removal (removes row corresponding to low firing neurons)


% 3. Extract parameters for classification
    % Start with data from 320ms, then iteratively add 20ms until 560ms for training
    binIndices = (startTime:binSize:endTime) / binSize; % 16 : 28
    intervalIdx = 1;

    for interval = binIndices % iteratively add testing time: 16, 17, 18, ... 28

    % 3.1 get firing rate data up to the certain time bin
        firingCurrent = zeros(numNeuronsNew*interval, numDirections*numTrials);
        for bin = 1 : interval
            firingCurrent(numNeuronsNew*(bin-1)+1:numNeuronsNew*bin, :) = firingData(numNeuronsNew*(bin-1)+1:numNeuronsNew*bin, :);
        end
    % Out: firingCurrent = firing rates data up to the current specified time interval (interval*95, 8*100)

    % 3.2 Principal Component Analysis for dimensionality reduction
        [eigenvectors, eigenvalues] = calcPCA(firingCurrent); % components = data projected onto the PCA axes
        % Out:
        %   eigenvectors: eigenvectors in ASCENDING order, NORMALISED
        %   eigenvalues: eigenvalues in ASCENDING order

        % Use variance explained to select how many components from PCA
        explained = sort(eigenvalues/sum(eigenvalues), 'descend'); % sort in descending order, variance explained
        pcaThreshold = 0.7; % threshold for selecting PCa components
        cumExplained = cumsum(explained);
        dimPCA = find(cumExplained >= pcaThreshold, 1, 'first');
        eigenvectors = eigenvectors(:, end-(dimPCA)+1:end); % components are in the order of ascending order so select from end
        % Out: eigenvectors: updated to only the top x dimensions determined by dimPCA

        % Reduce the dimensions of original data by projection onto the new dimensions
        pcaProjection = firingCurrent * eigenvectors; % e.g. (2660x800) * (800xdimPCA) = (2660xdimPCA)
        pcaProjection = pcaProjection./sqrt(sum(pcaProjection.^2)); % normalisation
    % Out: pcaProjection: projected firingCurrent data, reduced along the angle-trial axis, normalised
    
    % 3.3 Linear Discriminant Analysis
        dimLDA = 6;
        overallMean = mean(firingCurrent, 2); % (2660 x 1), mean rate for each neuron-bin

        % Extract the mean averages for each angle across all trials
        angleMeans = zeros(size(firingCurrent, 1), numDirections); % (2660 x 8)
        for angle = 1 : numDirections
           angleMeans(:, angle) = mean(firingCurrent(:, labels==angle), 2);
        end
        % Out: angleMeans: mean firing rates for each angle over 100 trials (numNeurons * numBins, 8)
        
        % Compute the scatter matrices
        S_B = (angleMeans - overallMean) * (angleMeans - overallMean)'; % (2660x8) * (8x2660) = (2660x2660)
        S_W = (firingCurrent - overallMean) * (firingCurrent - overallMean)' - S_B; % (2660x800) * (800x2660) - (2660x2660) = (2660x2660)

        % Compute eigen analysis to find directions maximising the S_B/S_W ratio
        % pcaProjection = weights we try to optimise to maximise the Fisher Criteron
        % The vector w that maximizes J(w) is used as the direction along which the data is projected to achieve the best class separability
        fisherCriterion = inv(pcaProjection' * S_W * pcaProjection) * (pcaProjection' * S_B * pcaProjection); % (dimPCA x dimPCA)
        [eigenvectors, eigenvalues] = eig(fisherCriterion);
        [~, sortIdx] = sort(diag(eigenvalues), 'descend');
        testProjection = pcaProjection * eigenvectors(:, sortIdx(1:dimLDA)); % LDA projection components for testing data (2660 x 6)
        trainProjected = testProjection' * (firingCurrent - overallMean); % training data projected onto LDA (6x2660) * (2660x800) = (6 x 800)
        
       % Store all the relevant weights for KNN
        modelParameters.knnClassify(intervalIdx).trainProjected = trainProjected; % (6 x 800) - 800 trials projected onto 6 components
        modelParameters.knnClassify(intervalIdx).testProjection = testProjection; % (2660 x 6)
        modelParameters.knnClassify(intervalIdx).dimPCA = dimPCA;
        modelParameters.knnClassify(intervalIdx).dimLDA = dimLDA;
        modelParameters.knnClassify(intervalIdx).meanFiring = overallMean; % (2660 x 1), mean rate for each neuron-bin
        intervalIdx = intervalIdx + 1;

    end % end of current interval


% 4. Principal Component Regression
    % we're edging so hard right now
    timeIntervals = startTime : binSize : endTime; % 320 : 20 : 560
    bins = repelem(binSize:binSize:endTime, numNeuronsNew); % time steps corresponding to the 28 bins replicated for 95 neurons
  
    % 4.1 Get the relevent position data
    handPosData = zeros(2*endTime/binSize, numTrials*numDirections);
    xPadded = zeros(numTrials, maxTimeSteps, numDirections);
    yPadded = zeros(numTrials, maxTimeSteps, numDirections);
    for angle = 1 : numDirections
        for trial = 1 : numTrials
            xPadded(trial, :, angle) = [trainingData(trial,angle).handPos(1,:), trainingData(trial,angle).handPos(1, end) * ones(1, maxTimeSteps - timeSteps(numTrials*(angle-1) + trial))];
            yPadded(trial, :, angle) = [trainingData(trial,angle).handPos(2,:), trainingData(trial,angle).handPos(2, end) * ones(1, maxTimeSteps - timeSteps(numTrials*(angle-1) + trial))]; 
            % padded data = 100 x 792 x 8 - padded with the last value
            
            % get average hand position data
            for bin = 1 : endTime/binSize  % 1 : 28
                handPosData(2*(bin-1)+1 : 2*bin, numTrials*(angle-1)+trial) = dataProcessed(trial, angle).handPos(:, bin);     
            end
            % handPosData = (28*2, 8*100)
        end
    end
    
    % 4.2 Compoute PCR coefficients
    for angle = 1: numDirections
        % select data for the specified angle
        xPos = handPosData((startTime/binSize * 2 - 1):2:end, labels==angle); % select the x pos for the selected angle (28x100), starting from 13th bin
        yPos = handPosData((startTime/binSize * 2):2:end, labels==angle); % (13 x 100), 13 bins from 320ms to 560ms

        for bin = 1: ((endTime-startTime)/binSize) + 1 % 1, 2, 3... 13 (320ms start)
            % select the hand position data at current bin, remove mean
            xCurrent = xPos(bin, :) - mean(xPos(bin, :)); % (1x100), mean removed
            yCurrent = yPos(bin, :) - mean(yPos(bin, :));

        % 4.1 PCA
            % select the firing data that corresponds to the iteratively increasing intervals and the given angle
            firingWindowed = firingData(bins <= timeIntervals(bin), labels == angle); % firing data for current interval, selected angle e.g. (2660x100)
            [eigenvectors, eigenvalues] = calcPCA(firingWindowed);

            % use variance explained to select how many components from PCA
            explained = sort(eigenvalues/sum(eigenvalues), 'descend'); % sort in descending order, variance explained
            cumExplained = cumsum(explained);
            dimPCA = find(cumExplained >= pcaThreshold, 1, 'first'); % threshold for selecting components is 80% variance
            eigenvectors = eigenvectors(:, end-(dimPCA)+1:end); % (100 x dimPCA)
            pcaProjection = firingWindowed * eigenvectors; % (2660x100) * (100xdimPCA) = (2660xdimPCA)

            % project windowed data onto the selected components 
            projection = pcaProjection' * (firingWindowed - mean(firingWindowed, 1)); % e.g. (dimPCAx2660) * (2660x100) = (dimPCAx100)
    
            % calculate regression coefficients - used to multiply with testing data
            xCoeff = (pcaProjection * inv(projection*projection') * projection) * xCurrent'; % (2660xdimPCA * (dimPCA x dimPCA) * (dimPCAx100)) * (100x1)
            yCoeff = (pcaProjection * inv(projection*projection') * projection) * yCurrent'; % = (2660x1)
    
            % record model parameters
            modelParameters.regression(angle, bin).xCoeff = xCoeff;
            modelParameters.regression(angle, bin).yCoeff = yCoeff;
            modelParameters.regression(angle, bin).firingMean = mean(firingWindowed, 1); % (1x800)
            modelParameters.positionMeans(bin).xMean = squeeze(mean(xPadded, 1)); % squeeze(mean(xPadded, 1)) = (792ms x 8), mean across 100 trials for each angle
            modelParameters.positionMeans(bin).yMean = squeeze(mean(yPadded, 1));
        end
    end % end of PCR


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
         
                % Initialisations
                spikeData = data(trial, angle).spikes; % extract spike data (98 x time steps)
                handPosData = data(trial, angle).handPos; % extract handPos data
                totalTime = size(spikeData, 2); % total number of time steps in ms
                binIndices = 1 : binSize : totalTime+1; % start of each time bin in ms
                spikeBins = zeros(numNeurons, length(binIndices)-1); % initialised binned spike data, (98 x number of bins)
                handPosBins = zeros(2, length(binIndices)-1); % initialised handPos data (2 directions x number of bins)

                % bin then squareroot the spike data            
                for bin = 1 : length(binIndices) - 1 % iterate through each bin
                    spikeBins(:, bin) = sum(spikeData(:, binIndices(bin):binIndices(bin+1)-1), 2); % sum spike number
                    handPosBins(:, bin) = mean(handPosData(1:2, binIndices(bin):binIndices(bin+1)-1), 2); % sample the hand position at the beginning of each bin
                end
                spikeBins = sqrt(spikeBins);

                % fill up the output
                dataProcessed(trial, angle).spikes = spikeBins; % spikes are now binned
                dataProcessed(trial, angle).handPos = handPosBins; % select only x and y
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


    function [eigenvectors, eigenvalues] = calcPCA(data)
    %----------------------------------------------------------------------
        % Manually computes the PCA (since toolboxes are BANNED)
    
        % Arguments:
        %   data: firing data

        % Return Value:
        %   eigenvectors: sorted in ASCENDING order
        %   eigenvalues: sorted in ASCENDING order
    %----------------------------------------------------------------------

        % Compute the eigenvectors and eigenvaues
        data0Mean = data - mean(data, 2);
        covMat = cov(data0Mean); % covariance matrix
        [eigenvectors, eigenvalues] = eig(covMat); % get eigenvalues and eigenvectors
        
        % Output processing
        eigenvalues = diag(eigenvalues); % only take the non-zero diagonal components, last one is largest
    
    end % end of calcPCA function

% Nested functions --------------------------------------------------------

end