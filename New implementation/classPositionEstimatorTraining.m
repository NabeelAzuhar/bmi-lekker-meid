% Last edit: 19/03/22
% Authors: Nabeel Azuhar Mohammed, Gloria Sun, Ioana Lazar, Alexia Badea
function [modelParameters] = classPositionEstimatorTraining(trainingData)
    
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
    count = 1;

% 1. Data Pre-processing + Filtering
    binSize = 20; % manually set bin size for data binning, 28 bins
    window = 50; % manually set window length for smoothing
    dataProcessed = dataProcessor(trainingData, binSize, window); % dataProcessed.rates = firing rates
    % Out: dataProcessed: binned (20ms) & smoothed spikes data, with .rates attribute

% 2. Find neurons with low firing rates for removal
    removed = {}; % data to filter out
 
    % make a matrix of all firing rate
    for angle = 1 : numDirections % each angle
        for trial = 1 : numTrials % each trial
            for bin = 1 : endTime/binSize % each time bin -> taking 28 time bins based on 560ms total time
                firingData(numNeurons*(bin-1)+1 : numNeurons*bin, numTrials*(angle-1)+trial) = dataProcessed(trial, angle).rates(:, bin);     
            end
        end
    end
    % Out: firingData: rows = 98 neurons * 28 bins, columns = 8 angles * 100 trials

    % take the average rate of each neuron across both dimensions, mark the ones with rates < 0.5
    lowFiringNeurons = []; % list to store the indices of low-firing neurons
    for neuron = 1 : numNeurons
        avgRate = mean(mean(firingData(neuron:98:end, :)));
        if avgRate < 0.5 % remove neurons with average rate less than 0.5
            lowFiringNeurons = [lowFiringNeurons, neuron];
        end
    end
    numNeuronsNew = numNeurons - length(lowFiringNeurons); % new number of neurons after removing the low firers
    removed{end+1} = lowFiringNeurons; % make sure the same neurons are removed in the test data
    modelParameters.lowFirers = removed; % record low firing neurons to model parameters output
    clear firingData % just in case
    
% 3. Extract parameters + Training
    % Start with data from 320ms, then iteratively add 20ms until 560ms for training
    binIndices = (startTime:binSize:endTime) / binSize; % a list of all the bins that are iteratively added for testing
    for interval = binIndices % iteratively add testing time (add 20ms every iteration)

    % 3.1 get firing rate data up to the certain time frame
        % firingData is the unfiltered firing rate data of all neurons at all time bins
        % firingCurrent is the filtered firing rate data including only non-lowfirers for the selected time interval
        for angle = 1: numDirections
            for trial = 1: numTrials
                for bin = 1: interval % only taking bins up to the current allowed time (320:20:560 ms)
                    % firingCurrent is the firing rates up to the current interval
                    firingCurrent(numNeurons*(bin-1)+1 : numNeurons*bin, numTrials*(angle-1)+trial) = dataProcessed(trial, angle).rates(:, bin);     
                end
            end
        end
        % Out: firingCurrent = firing rates data up to the current specified time interval

        % remove the low firing neurons from current firing data
        removedRows = []; % rows of data to remove (for low firing neurons at all its time bins)
        for lowFirer = lowFiringNeurons
            removedRows = [removedRows, lowFirer:numNeurons:length(firingCurrent)];
        end
        firingCurrent(removedRows, :) = []; % remove the rows for the low firers
        % NOW have numNeuronNew as the number of neurons! (95)

    % 3.2 Principal Component Analysis for dimensionality reduction
        [components, eigenvalues] = calcPCA(firingCurrent); % components = data projected onto the PCA axes
        % Out:
        %   components: projections of data onto eigenvectors, ASCENDING
        %   eigenvalues: eigenvalues in ASCENDING order

        % use variance explained to select how many components from PCA
        explained = sort(eigenvalues/sum(eigenvalues), 'descend'); % sort in descending order, variance explained
        cumExplained = cumsum(explained);
        dimPCA = find(cumExplained >= 0.7, 1, 'first'); % threshold for selecting components is 80% variance
        components = components(:, end-(dimPCA):end); % components are in the order of ascending order so select from end
        
    % 3.3 Linear Discriminant Analysis
        dimLDA = 6; % numbers of dimensions to pick for , arbitrary for now
        
        % tmp = matrix to store average firing data for each angle
        tmp = zeros(size(firingCurrent, 1), numDirections); % temporary place holder, each column = an angle
        for angle = 1: numDirections
            tmp(:, angle) =  mean(firingCurrent(:, numTrials*(angle-1)+1 : angle*numTrials), 2); % taking the mean of each angle across 100 trials
        end
        % Out: rows = numNeurons * 28 time bins, columns = 8 angles

        scatterBetween = (tmp - mean(firingCurrent, 2)) * (tmp - mean(firingCurrent, 2))'; % between-class scatter matrix
        scatterOverall =  (firingCurrent - mean(firingCurrent,2)) * (firingCurrent - mean(firingCurrent,2))';
        scatterWithin = scatterOverall - scatterBetween;
        
        [eigVectorsLDA, eigValuesLDA] = eig(((components' * scatterWithin * components)^-1 ) * (components' * scatterBetween * components));
        [~, sortIdx] = sort(diag(eigValuesLDA), 'descend');
        optWeights = components * eigVectorsLDA(:, sortIdx(1:dimLDA)); % optimum parameters
        wLDA = optWeights' * (firingCurrent - mean(firingCurrent, 2)); % optimum projection

    % 3.4 Store all the relevant weights for KNN
        modelParameters.classify(count).wLDA_kNN = wLDA;
        modelParameters.classify(count).dPCA_kNN = dimPCA;
        modelParameters.classify(count).dLDA_kNN = dimLDA;
        modelParameters.classify(count).wOpt_kNN = optWeights;
        modelParameters.classify(count).mFire_kNN = mean(firingCurrent, 2);
        count = count + 1;

    end % end of the selected training interval

% 4. Principal Component Regression 
    % xn, yn = x, y position data padded with the last position value of each trial
    % x, y = SAMPLED x, y position data using the set bin size (20ms)
    timeIntervals = startTime : binSize : endTime; % 320 : 20 : 560
    [xn, yn, x, y] = positionSampled(trainingData, numDirections, numTrials, binSize);
    xTest = x(:, (timeIntervals)/binSize, :); % select the relevant bins from the sampled data
    yTest = y(:, (timeIntervals)/binSize, :);
    bins = repelem(binSize:binSize:endTime, numNeuronsNew); % time steps corresponding to the 28 bins replicated for 8 neurons

    directionLabels = [1 * ones(1, numTrials), ...
             2 * ones(1, numTrials), ...
             3 * ones(1, numTrials), ...
             4 * ones(1, numTrials), ...
             5 * ones(1, numTrials), ...
             6 * ones(1, numTrials), ...
             7 * ones(1, numTrials), ...
             8 * ones(1, numTrials)];
    
    for angle = 1: numDirections
        % get the spikes data for each direction
        xDirection = squeeze(xTest(:, :, angle));
        yDirection = squeeze(yTest(:, :, angle));
        
        for bin = 1: ((endTime-startTime)/binSize) + 1 % go through each testing time index
            % mean removal for PCR
            xPCR = xDirection(:, bin) - mean(xDirection(:, bin));
            yPCR = yDirection(:, bin) - mean(yDirection(:, bin));
            
        % 4.1 PCA
            % select the firing data that corresponds to the iteratively increasing intervals and the given angle
            firingWindowed = firingCurrent(bins <= timeIntervals(bin), directionLabels == angle);
            [components , eigenvalues] = calcPCA(firingWindowed);

            % use variance explained to select how many components from PCA
            explained = sort(eigenvalues/sum(eigenvalues), 'descend'); % sort in descending order, variance explained
            cumExplained = cumsum(explained);
            dimPCA = find(cumExplained >= 0.7, 1, 'first'); % threshold for selecting components is 80% variance
            components = components(:, end-(dimPCA):end); % components are in the order of ascending order so select from end

            % project windowed data onto the selected components 
            projection = components' * (firingWindowed - mean(firingWindowed, 1));

            % calculate regression coefficients
            xCoeff = (components * inv(projection*projection') * projection) * xPCR;
            yCoeff = (components * inv(projection*projection') * projection) * yPCR;

            % record model parameters
            modelParameters.pcr(angle, bin).xM = xCoeff;
            modelParameters.pcr(angle, bin).yM = yCoeff;
            modelParameters.pcr(angle, bin).fMean = mean(firingWindowed, 1);
            modelParameters.averages(bin).xMean = squeeze(mean(xn, 1));
            modelParameters.averages(bin).yMean = squeeze(mean(yn, 1));
        end
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
        data0Mean = data - mean(data, 2); % mean removal across trials (features) 
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