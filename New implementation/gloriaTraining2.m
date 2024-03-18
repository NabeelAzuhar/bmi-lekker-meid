function [modelParameters] = gloriaTraining2(trainingData)
    
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
%------------------

% 1. Initialisations
    modelParameters = struct; % output
    numNeurons = size(trainingData(1, 1).spikes, 1);
    numTrials = size(trainingData, 1);
    numDirections = size(trainingData, 2);
    startTime = 320; % set by the competition rules
    binSize = 20; % manually set bin size for data binning - 20ms
    window = 50; % manually set window length for smoothing - 50ms

    % Determine max and min number of time steps
    timeSteps = [];
    for angle = 1 : numDirections
        for trial = 1 : numTrials
            timeSteps = [timeSteps, size(trainingData(trial, angle).spikes, 2)];
        end
    end
    maxTimeSteps = max(timeSteps); % take the longest time length across all trials
    endTime = floor(min(timeSteps)/binSize) * binSize; % smallest time length in training data floored to time bin of 20ms

    % Labels for selecting data for a selected angle from data
    labels = repmat(1:numDirections, numTrials, 1);
    labels = labels(:); % (800 x 1) - identifies reaching angle of every 100 trials


% 2. Data Pre-processing + Filtering
    dataProcessed = dataProcessor(trainingData, binSize, window); % dataProcessed.rates = firing rates
    % Out: dataProcessed: binned (20ms) & smoothed spikes data, with .rates attribute & binned x, y handPos as .handPos
    
    % 2.1 Find neurons with low firing rates for removal
    % Fetch firing rate data and create matrix
    for angle = 1 : numDirections % each angle
        for trial = 1 : numTrials % each trial
            for bin = 1 : endTime/binSize % each time bin -> 1 : 28
                firingData(numTrials*(angle-1)+trial, numNeurons*(bin-1)+1 : numNeurons*bin) = dataProcessed(trial, angle).rates(:, bin)';     
            end
        end
    end

    % Mark the neurons to be removed (with rates < 0.5)
    lowFiringNeurons = []; % list to store the indices of low-firing neurons
    for neuron = 1 : numNeurons
        avgRate = mean(mean(firingData(:, neuron:98:end)));
        if avgRate < 0.5 % remove neurons with average rate less than 0.5
            lowFiringNeurons = [lowFiringNeurons, neuron];
        end
    end
    modelParameters.lowFirers = lowFiringNeurons; % record low firing neurons to model parameters output
    % Out: lowFiringNeurons: a list containing all the low-firing neurons

    % Remove the neurons
    removedCols = []; % rows of data to remove (for low firing neurons at all its time bins)
    for lowFirer = lowFiringNeurons
        removedCols = [removedCols, lowFirer:numNeurons:length(firingData)];
    end
    firingData(:, removedCols) = []; % remove the rows for the low firers
    numNeuronsNew = numNeurons - length(lowFiringNeurons); % new number of neurons after removing the low firers
    % Out:
    %   numNeuronsNew: updated number of neurons after rate filtering
    %   firingData: filtered rows after neuron removal (removes row corresponding to low firing neurons)
   

% 3. Extract parameters for classification
    binIndices = (startTime:binSize:endTime) / binSize; % 16, 17, 18, ... 28
    intervalIdx = 1; % shift the 16th bin to be the 1st

    % Start with data from 320ms, then iteratively add 20ms until 560ms for training
    for interval = binIndices % 16 : 28

    % 3.1 Get firing rate up to the selected time bin
        firingCurrent = zeros(numDirections*numTrials, numNeuronsNew*interval); % e.g. (800 x 2660)
        for bin = 1 : interval
            firingCurrent(:, numNeuronsNew*(bin-1)+1:numNeuronsNew*bin) = firingData(:, numNeuronsNew*(bin-1)+1:numNeuronsNew*bin);
        end
    % Out: firingCurrent = firing rates data up to the current specified time interval (interval*95, 8*100)
    
    % 3.2 Principal Component Analysis for dimensionality reduction
        [eigenvectors, eigenvalues] = calcPCA(firingCurrent); % components = data projected onto the PCA axes
        % PCA reducing dimensions along the neuron-bin axis
        % Out:
        %   eigenvectors: eigenvectors in ASCENDING order (1520 x 1520)
        %   eigenvalues: eigenvalues in ASCENDING order
        
        % Use variance explained to select how many components from PCA
        explained = sort(eigenvalues/sum(eigenvalues), 'descend'); % sort in descending order, variance explained
        pcaThreshold = 0.7;
        cumExplained = cumsum(explained);
        dimPCA = find(cumExplained >= pcaThreshold, 1, 'first'); % threshold for selecting components is 80% variance
        eigenvectors = eigenvectors(:, end-(dimPCA)+1:end); % components are in the order of ascending order so select from end
        % Out: eigenvectors: updated to only the top x dimensions determined by dimPCA
        
        % Reduce the dimensions of original data by projection onto the new dimensions
        pcaProjection = firingCurrent * eigenvectors; % e.g. (800x2660) * (2660 x dimPCA) = (800 x dimPCA)
        pcaProjection = pcaProjection./sqrt(sum(pcaProjection.^2)); % normalisation
    % Out: pcaProjection: projected firingCurrent data, reduced along the neuron-bin axis
    
    % 3.3 Linear Discriminant Analysis
        dimLDA = 6;
        overallMean = mean(pcaProjection, 1); % (1 x dimPCA), meaning rate for each feature
    
        % Extract the mean averages for each angle across all trials
        angleMeans = zeros(numDirections, size(pcaProjection, 2));
        for angle = 1 : numDirections
           angleMeans(angle, :) = mean(pcaProjection(labels==angle, :), 1); % (8 x dimPCA)
        end
        % Out: angleMeans: mean firing rates for each angle over 100 trials

        % Compute the scatter matrices
        S_B = (angleMeans - overallMean)' * (angleMeans - overallMean); % (dimPCA x dimPCA)
        S_W = (pcaProjection - overallMean)' * (pcaProjection- overallMean) - S_B; % (dimPCA x dimPCA)
        
        % Eigenanalysis of S_B/S_W
        [eigenvectors, eigenvalues] = eig(S_B * S_W^-1);
        [~, sortIdx] = sort(diag(eigenvalues), 'descend');
        eigenvectors = eigenvectors(:, sortIdx(1:dimLDA)); % (dimPCA x dimLDA)
        pcaLdaProjection = pcaProjection * eigenvectors; % optimum discriminant directions (800 x 6)
        firingProjection = pcaLdaProjection' * (firingCurrent - mean(firingCurrent, 1)); % (6 x 2660)

       % Store all the relevant weights for KNN
        modelParameters.classify(intervalIdx).wLDA_kNN = firingProjection;
        modelParameters.classify(intervalIdx).wOpt_kNN = pcaLdaProjection;
        modelParameters.classify(intervalIdx).dPCA_kNN = dimPCA;
        modelParameters.classify(intervalIdx).dLDA_kNN = dimLDA;
        modelParameters.classify(intervalIdx).mFire_kNN = mean(firingCurrent, 1);
        intervalIdx = intervalIdx + 1;

    end % end of selected interval


% 4. Principal Component Regression 
    timeIntervals = startTime : binSize : endTime; % 320 : 20 : 560
    bins = repelem(binSize:binSize:endTime, numNeuronsNew); % time steps corresponding to the 28 bins replicated for 95 neurons

    % 4.1 Get sampled x and y position data starting from 0 to 560ms
    for angle = 1 : numDirections
        for trial = 1 : numTrials
            xPadded(trial, :, angle) = [trainingData(trial,angle).handPos(1,:), trainingData(trial,angle).handPos(1, end) * ones(1, maxTimeSteps - timeSteps(numTrials*(angle-1) + trial))];
            yPadded(trial, :, angle) = [trainingData(trial,angle).handPos(2,:), trainingData(trial,angle).handPos(2, end) * ones(1, maxTimeSteps - timeSteps(numTrials*(angle-1) + trial))]; 
            % 100 x 792 x 8

            for bin = 1 : endTime/binSize  % 1 : 28
                handPosData(2*(bin-1)+1 : 2*bin, numTrials*(angle-1)+trial) = dataProcessed(trial, angle).handPos(:, bin);     
            end
        end
    end
    % Out: handPosData = (28*2, 8*100)
    
    for angle = 1: numDirections
        xPos = handPosData((startTime/binSize * 2 - 1):2:end, labels==angle); % select the x pos for the selected angle (28x100), starting from 13th bin
        yPos = handPosData((startTime/binSize * 2):2:end, labels==angle); % (13 x 100), 13 bins from 320ms to 560ms

        for bin = 1: ((endTime-startTime)/binSize) + 1 % 1, 2, 3... 13 (320ms start)
            % select the hand position data at current bin, remove mean
            xPCR = xPos(bin, :) - mean(xPos(bin, :)); % (1x100)
            yPCR = yPos(bin, :) - mean(yPos(bin, :));

        % 4.1 PCA
            % select the firing data that corresponds to the iteratively increasing intervals and the given angle
            firingDataPCR = firingData'; % transform just for PCR
            firingWindowed = firingDataPCR(bins <= timeIntervals(bin), labels == angle); % firing data for current interval, selected angle e.g. (2660x100)
            [eigenvectors, eigenvalues] = calcPCA(firingWindowed); % reduced along the neuron-bin dimension (column), eigenvectors = (100 x 100)

            % use variance explained to select how many components from PCA
            explained = sort(eigenvalues/sum(eigenvalues), 'descend'); % sort in descending order, variance explained
            cumExplained = cumsum(explained);
            dimPCA = find(cumExplained >= pcaThreshold, 1, 'first'); % threshold for selecting components is 80% variance
            eigenvectors = eigenvectors(:, end-(dimPCA)+1:end); % (100 x dimPCA)
            pcaProjection = firingWindowed * eigenvectors; % (2660x100) * (100xdimPCA) = (2660xdimPCA)
            pcaProjection = pcaProjection./sqrt(sum(pcaProjection.^2)); % normalisation

            % project windowed data onto the selected components 
            projection = pcaProjection' * (firingWindowed - mean(firingWindowed, 1)); % e.g. (dimPCAx2660) * (2660x100) = (dimPCAx100)
    
            % calculate regression coefficients - used to multiply with firing data in testing
            % firing data projected onto pca components, then projected onto hand position
            xCoeff = (pcaProjection * inv(projection*projection') * projection) * xPCR'; % (2660xdimPCA * (dimPCA x dimPCA) * (dimPCAx100)) * (100x1)
            yCoeff = (pcaProjection * inv(projection*projection') * projection) * yPCR'; % = (2660x1)
    
            % record model parameters
            modelParameters.pcr(angle, bin).xM = xCoeff;
            modelParameters.pcr(angle, bin).yM = yCoeff;
            modelParameters.pcr(angle, bin).fMean = mean(firingWindowed, 1);
            modelParameters.averages(bin).xMean = squeeze(mean(xPadded, 1)); % squeeze(mean(xPadded, 1)) = (792ms x 8), mean across 100 trials for each angle
            modelParameters.averages(bin).yMean = squeeze(mean(yPadded, 1));
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
                    handPosBins(:, bin) = handPosData(1:2, binIndices(bin)); % sample the hand position at the beginning of each bin
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
        data0Mean = data - mean(data, 1);
        covMat = cov(data0Mean); % covariance matrix
        [eigenvectors, eigenvalues] = eig(covMat); % get eigenvalues and eigenvectors
        
        % Output processing
        eigenvalues = diag(eigenvalues); % only take the non-zero diagonal components, last one is largest
    
    end % end of calcPCA function


% Nested functions --------------------------------------------------------
    

end % end of function