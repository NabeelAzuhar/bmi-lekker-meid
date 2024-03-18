function [fisherCriterion] = gloriaTraining(trainingData)
    
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

% Initialisations
    modelParameters = struct; % output
    numNeurons = size(trainingData(1, 1).spikes, 1); % stays constant throughout
    numTrials = size(trainingData, 1);
    numDirections = size(trainingData, 2);
    startTime = 320;
    endTime = 560; % smallest time length in training data rounded to time bin of 20ms
    labels = repmat(1:numDirections, numTrials, 1); % labels for selecting data for a selected angle
    labels = labels(:);

% 1. Data Pre-processing + Filtering
    binSize = 20; % manually set bin size for data binning, 28 bins
    window = 50; % manually set window length for smoothing
    dataProcessed = dataProcessor(trainingData, binSize, window); % dataProcessed.rates = firing rates
    % Out: dataProcessed: binned (20ms) & smoothed spikes data, with .rates attribute

% 2. Find neurons with low firing rates for removal
    % 2.1 Fetch firing rate data and create matrix
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
    binIndices = (startTime:binSize:endTime) / binSize; % 16, 17, 18, ... 28
    intervalIdx = 1;

    for interval = binIndices % iteratively add testing time (add 20ms every iteration)

    % 3.1 get firing rate data up to the certain time bin
        firingCurrent = zeros(numNeuronsNew*interval, numDirections*numTrials);
        for bin = 1 : interval
            firingCurrent(numNeuronsNew*(bin-1)+1:numNeuronsNew*bin, :) = firingData(numNeuronsNew*(bin-1)+1:numNeuronsNew*bin, :);
        end
    % Out: firingCurrent = firing rates data up to the current specified time interval

    % 3.2 Principal Component Analysis for dimensionality reduction
        [eigenvectors, eigenvalues] = calcPCA(firingCurrent); % components = data projected onto the PCA axes
        % Out:
        %   eigenvectors: eigenvectors in ASCENDING order, NORMALISED
        %   eigenvalues: eigenvalues in ASCENDING order

        % Use variance explained to select how many components from PCA
        explained = sort(eigenvalues/sum(eigenvalues), 'descend'); % sort in descending order, variance explained
        pcaThreshold = 0.7;
        cumExplained = cumsum(explained);
        dimPCA = find(cumExplained >= pcaThreshold, 1, 'first'); % threshold for selecting components is 80% variance
        eigenvectors = eigenvectors(:, end-(dimPCA)+1:end); % components are in the order of ascending order so select from end
        % Out: eigenvectors: updated to only the top x dimensions determined by dimPCA

        % Reduce the dimensions of original data by projection onto the new dimensions
        pcaProjection = firingCurrent * eigenvectors; % e.g. (2660x800) * (800x10) = (2660x10)
    % Out: pcaProjection: projected firingCurrent data, reduced along the angle-trial axis
    
    % 3.3 Linear Discriminant Analysis
        dimLDA = 6;
        overallMean = mean(firingCurrent, 2); % (2660 x 1), meaning rate for each neuron-bin

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
        fisherCriterion = ((pcaProjection' * S_W * pcaProjection)^-1) * (pcaProjection' * S_B * pcaProjection);
        

    end % end of current interval



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
        eigenvectors = eigenvectors./sqrt(sum(eigenvectors.^2)); % normalisation
    
    end % end of calcPCA function


    function [xn, yn, x, y] = positionSampled(data, numDirections, numTrials, binSize)
    %----------------------------------------------------------------------
        % Arguments:
        %   data: training data
        %   numDirections = number of different reaching angles
        %   numTrain = number of training samples used 
        %   binSize = binning resolution of the spiking/rate data

        % Return Value:
        %   xn
        %   yn
        %   x
        %   y   
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

end