% Last edit: 19/03/22
% Authors: Nabeel Azuhar Mohammed, Gloria Sun, Ioana Lazar, Alexia Badea
function [modelParameters] = gloriaPositionEstimatorTraining(trainingData)
    
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
    startTime = 320;
    endTime = 560; % smallest time length in training data rounded to time bin of 20ms
    labels = repmat(1 : numDirections, numTrials, 1); % Creates label vector that identifies the reaching angle for each row
    labels = labels(:);  % Ensure it's a column vector, resulting in a 12800x1 vector
    
% 1. Data Pre-processing + Filtering
    binSize = 20; % manually set bin size for data binning, 28 bins
    window = 50; % manually set window length for smoothing
    dataProcessed = dataProcessor(trainingData, binSize, window); % dataProcessed.rates = firing rates
    % Out: dataProcessed: binned (20ms) & smoothed spikes data, with .rates attribute

    % 1.1. Find neurons with low firing rates for removal
    % make a matrix of all firing rate, limit to 28 bins
    for angle = 1 : numDirections % each angle
        for trial = 1 : numTrials % each trial
            for bin = 1 : endTime/binSize % each time bin -> taking 28 time bins based on 560ms total time
                firingData(numTrials*(angle-1)+trial, numNeurons*(bin-1)+1 : numNeurons*bin) = dataProcessed(trial, angle).rates(:, bin)';     
            end
        end
    end
    % identify low firing neurons
    lowFiringNeurons = []; 
    for neuron = 1 : numNeurons
        avgRate = mean(mean(firingData(:, neuron:98:end)));
        if avgRate < 0.5 % remove neurons with average rate less than 0.5
            lowFiringNeurons = [lowFiringNeurons, neuron];
        end
    end

    % remove low firing neurons
    removedCols = []; % rows of data to remove (for low firing neurons at all its time bins)
    for lowFirer = lowFiringNeurons
        removedCols = [lowFirer:numNeurons:length(firingData), removedCols];
    end
    firingData(:, removedCols) = []; % remove the rows for the low firers
    modelParameters.lowFirers = lowFiringNeurons;
    numNeuronsNew = numNeurons - length(lowFiringNeurons); % new number of neurons after rate filtering
% Out: firingData: rows = 95 neurons * 28 bins, columns = 8 angles * 100 trials

% 2. Sample hand position data for PCR
    binIndices = (startTime:binSize:endTime) / binSize; % [16, 17, 18 ... 28]
    [xPadded, yPadded, xBinned, yBinned] = positionSampled(trainingData, numDirections, numTrials, binSize);
    % get hand position at the bins between start and end times
    xInterval = xBinned(:, binIndices, :); % now 1st bin corresponds to the 16th from start time
    yInterval = yBinned(:, binIndices, :);

% 3. Extract parameters + Training
    % Start with data from 320ms, then iteratively add 20ms until 560ms 
    intervalIndex = 1; % interval but starts from 1 (16th bin intialised to be 1)

    for interval = binIndices % iteratively add testing time (add 20ms every iteration)

    % 3.1 Get firing rate data up to the current time interval
        intervalArray = zeros(numDirections*numTrials, interval*numNeuronsNew);
        for bin = 1: interval % only taking bins up to the current allowed time (320:20:560 ms)
            intervalArray(:, numNeuronsNew*(bin-1)+1 : numNeuronsNew*bin) = firingData(:,  numNeuronsNew*(bin-1)+1 : numNeuronsNew*bin);
        end
        % Out: intervalArray contain the firing data from 0ms up to the current time bin

    % 3.2 Principal Component Analysis for dimensionality reduction
        [componentsPCA, eigenvalues] = calcPCA(intervalArray); % components = data projected onto the PCA axes
        pcaThreshold = 0.7;

        % use variance explained to select how many components from PCA
        explained = sort(eigenvalues/sum(eigenvalues), 'descend'); % sort in descending order, variance explained
        cumExplained = cumsum(explained);
        dimPCA = find(cumExplained >= pcaThreshold, 1, 'first'); % threshold for selecting components is 80% variance
        componentsPCA = componentsPCA(:, end-(dimPCA)+1:end); % components are in the order of ascending order so select from end
    
    % 3.3 Linear Discriminant Analysis for classification
        
        % initialisations
        overallMean = mean(componentsPCA, 1);
        S_W = zeros(size(componentsPCA,2), size(componentsPCA,2)); % Within-class scatter matrix
        S_B = zeros(size(componentsPCA,2), size(componentsPCA,2)); % Between-class scatter matrix
        
        % Calculate means for each reaching angle
        for angle = 1:numDirections
            angleData = componentsPCA(labels == angle, :); % select trial data corresponding to the selected angle
            angleMean = mean(angleData, 1); % mean across the 100 trials for each feature
            angleData = angleData - angleMean; % centre angle data

            % within-class scatter
            S_W = S_W + (angleData' * angleData); % accumuate to S_W

            % between-class scatter
            meanDiff = angleMean - overallMean;
            S_B = S_B + (numTrials * (meanDiff' * meanDiff)); % Update S_B
        end

        % eigenanalysis to look for discriminants (LDA components)
        dimLDA = 6; % explains 0.999 variances
        [eigVectors, ~] = eig((S_W^-1 ) * S_B);
        componentsLDA = eigVectors(:, end-(dimLDA)+1:end); % dimLDA largest components
        pcaLdaProjection = componentsPCA * componentsLDA; % PCA data projected onto LDA components
        firingProjection = pcaLdaProjection' * (intervalArray - mean(intervalArray, 1));

        % store parameters for classification
        modelParameters.classify(intervalIndex).wOpt_kNN = pcaLdaProjection;
        modelParameters.classify(intervalIndex).wLDA_kNN = firingProjection;
        modelParameters.classify(intervalIndex).dPCA_kNN = dimPCA;
        modelParameters.classify(intervalIndex).dLDA_kNN = dimLDA;
        modelParameters.classify(intervalIndex).mFire_kNN = mean(intervalArray, 1);
        intervalIndex = intervalIndex + 1;
    end % end of each interval
    
% % 4. Principal Component Regression
%     timeIntervals = startTime : binSize : endTime;
%     bins = repelem(binSize:binSize:endTime, numNeuronsNew); % time steps corresponding to the 28 bins replicated for 8 neurons
% 
%     % create coefficients for each direction
%     for angle = 1 : numDirections
%         % get hand position at current angle
%         xDirection = squeeze(xInterval(:, :, angle));
%         yDirection = squeeze(yInterval(:, :, angle));
% 
%         for bin = 1 : intervalIndex-1 % 1: 14 (1 corresponds to start time 320ms)
%             % zero mean position data at selected time bin
%             xPCR = xDirection(:, bin) - mean(xDirection(:, bin));
%             yPCR = yDirection(:, bin) - mean(yDirection(:, bin));
% 
%         % 4.1 PCA
%             % select the firing data that corresponds to the iteratively increasing intervals and the given angle
%             firingWindowed = firingData(labels == angle, bins <= timeIntervals(bin));
%             [components , eigenvalues] = calcPCA(firingWindowed);
%         end
%     end
 


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
                for bin = 1 : numel(binStarts) - 1 % iterate through each bin
                    spikeBins(:, bin) = sum(spikeData(:, binStarts(bin):binStarts(bin+1)-1), 2); % sum spike number
                end
                spikeBins = sqrt(spikeBins);

                % fill up the output
                dataProcessed(trial, angle).spikes = spikeBins; % spikes are now binned
                dataProcessed(trial, angle).handPos = data(trial, angle).handPos(1:2, :); % select only x and y
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


    function [components, eigenvalues] = calcPCA(data)
    %----------------------------------------------------------------------
        % Manually computes the PCA (since toolboxes are BANNED)

        % Arguments:
        %   data: firing data

        % Return Value:
        %   eigenvectors: sorted in ASCENDING order
        %   eigenvalues: sorted in ASCENDING order
    %----------------------------------------------------------------------

        % Compute the eigenvectors and eigenvaues
        data0Mean = data - mean(data, 1); % mean removal across features
        covMat = cov(data0Mean); % covariance matrix
        [eigenvectors, eigenvalues] = eig(covMat); % get eigenvalues and eigenvectors
        
        % Output processing
        eigenvalues = diag(eigenvalues); % only take the non-zero diagonal components, last one is largest
        components = data * eigenvectors; % project firing rate data onto eigenvectors
        components = components./sqrt(sum(components.^2)); % normalisation
    
    end % end of calcPCA function


    function [xn, yn, x, y] = positionSampled(data, numDirections, numTrials, binSize)
    %----------------------------------------------------------------------
        % Arguments:
        %   data: training data
        %   numDirections = number of different reaching angles
        %   numTrain = number of training samples used 
        %   binSize = binning resolution of the spiking/rate data

        % Return Value:
        %   xn: original x position data padded with the last value
        %   yn: original y position data padded with the last value
        %   x: sampled x position data according to binSize
        %   y: sampled y position data according to binSize   
    %----------------------------------------------------------------------

        % Find the maximum trajectory
        trialCell = struct2cell(data);
        timeSteps = [];
        for spikeData = 2 : 3 : numTrials * numDirections * 3 % 100 * 8 trials each with 3 fields, only locating spikes data
            timeSteps = [timeSteps, size(trialCell{spikeData},2)]; % append the lengths of all trajectory
        end
        maxTimeSteps = max(timeSteps); % maximum trial length
    
        % Initialise position matrices for both x and y position data
        xn = zeros([numTrials, maxTimeSteps, numDirections]); % 3D matrix
        yn = xn; % copy x for y
    
        for angle = 1: numDirections
            for trial = 1:numTrials
                
                % Padding position data with the last value
                xn(trial, :, angle) = [data(trial,angle).handPos(1,:), data(trial,angle).handPos(1, end) * ones(1, maxTimeSteps - timeSteps(numTrials*(angle-1) + trial))];
                yn(trial, :, angle) = [data(trial,angle).handPos(2,:), data(trial,angle).handPos(2, end) * ones(1, maxTimeSteps - timeSteps(numTrials*(angle-1) + trial))];  
                
                % Resampling x and y according to the binning size
                tmpX = xn(trial, :, angle);
                tmpY = xn(trial, :, angle);
                x(trial, :, angle) = tmpX(1:binSize:end); % sample the position value at every binSize
                y(trial, :, angle) = tmpY(1:binSize:end);
            end
        end
    end
% Nested functions --------------------------------------------------------

end