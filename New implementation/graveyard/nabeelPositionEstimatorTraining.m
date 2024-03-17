% Last edit: 19/03/22
% Authors: Nabeel Azuhar Mohammed, Gloria Sun, Ioana Lazar, Alexia Badea
function xDirection = nabeelPositionEstimatorTraining(trainingData)
    
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
    intervalIndex = 1;

% 1. Data Pre-processing + Filtering
    binSize = 20; % manually set bin size for data binning, 28 bins
    window = 50; % manually set window length for smoothing
    dataProcessed = dataProcessor(trainingData, binSize, window); % dataProcessed.rates = firing rates
    % Out: dataProcessed: binned (20ms) & smoothed spikes data, with .rates attribute

% 2. Create matrix suitable for PCA
    % make a matrix of all firing rate
    for angle = 1 : numDirections % each angle
        for trial = 1 : numTrials % each trial
            for bin = 1 : endTime/binSize % each time bin -> taking 28 time bins based on 560ms total time
                firingData(((angle - 1) * numTrials * (endTime/binSize)) + ((trial - 1) * (endTime/binSize)) + bin, :) = dataProcessed(trial, angle).rates(:, bin)';     
            end
        end
    end
    % Out: firingData: rows = 8 angles * 100 trials * 28 bins, columns = 98 neuron

% 3. Remove low firing neurons
    lowFiringNeurons = [];
    for neuron = 1 : numNeurons
        avgRate = mean(mean(firingData(:, neuron)));
        if avgRate < 0.5 % remove neurons with average rate less than 0.5
            lowFiringNeurons = [lowFiringNeurons, neuron];
        end
    end
    modelParameters.lowFirers = lowFiringNeurons;
    firingData(:, lowFiringNeurons) = [];
    numNeuronsNew = size(firingData, 2);

% 4. Initialise data for PCR at the end
    % xn, yn = x, y position data padded with the last position value of each trial
    % x, y = SAMPLED x, y position data using the set bin size (20ms)
    timeIntervals = startTime : binSize : endTime; % 320 : 20 : 560
    [xn, yn, x, y] = positionSampled(trainingData, numDirections, numTrials, binSize);
    xTest = x(:, (timeIntervals)/binSize, :); % select the relevant bins from the sampled data
    yTest = y(:, (timeIntervals)/binSize, :);

% 5. Extract parameters + Training
    % Start with data from 320ms, then iteratively add 20ms until 560ms for training
    binIndices = (startTime:binSize:endTime) / binSize; % a list of all the bins that are iteratively added for testing
    
    % 5.1 Get firing rate data up to the current time interval
    for interval = binIndices % iteratively add testing time (add 20ms every iteration)
        intervalArray = [];
        for i = 1 : interval + ((endTime / binSize) - interval) : size(firingData, 1) - interval + 1
            temp = firingData(i:i+interval-1, :);
            intervalArray = [intervalArray; temp];
        end
        % intervalArray will contain the first x number of time bins for further analysis

    % 5.2 Principal Component Analysis for dimensionality reduction
        [componentsPCA, eigenvalues] = calcPCA(intervalArray); % components = data projected onto the PCA axes
        % Out:
        %   components: projections of data onto eigenvectors, ASCENDING
        %   eigenvalues: eigenvalues in ASCENDING order

        % use variance explained to select how many components from PCA
        explained = sort(eigenvalues/sum(eigenvalues), 'descend'); % sort in descending order, variance explained
        cumExplained = cumsum(explained);
        PCAThreshold = 0.7; % threshold for selecting components is 70% variance
        dimPCA = find(cumExplained >= PCAThreshold, 1, 'first'); 
        componentsPCA = componentsPCA(:, end-(dimPCA)+1:end); % components are in the order of ascending order so select from end

    % 5.3 Linear Discriminant Analysis
        % Creates label vector that identifies the reaching angle for each row
        labels = repmat(1 : numDirections, numTrials * interval, 1);
        labels = labels(:);  % Ensure it's a column vector, resulting in a 12800x1 vector

        % Assume X is your PCA-reduced dataset (12800 rows x dimPCA columns)
        overallMean = mean(componentsPCA, 1); % 1 x dimPCA vector
        S_W = zeros(dimPCA, dimPCA); % Within-class scatter matrix
        S_B = zeros(dimPCA, dimPCA); % Between-class scatter matrix
        
        % Calculate means for each reaching angle
        angleMeans = zeros(numDirections, dimPCA); % Preallocate for efficiency
        for angle = 1:numDirections
            angleData = componentsPCA(labels == angle, :);
            angleMeans(angle, :) = mean(angleData, 1);
            deviations = angleData - angleMeans(angle, :);
            S_W = S_W + deviations' * deviations; % Update S_W
            meanDiff = angleMeans(angle, :) - overallMean;
            S_B = S_B + (numTrials * interval) * (meanDiff' * meanDiff); % Update S_B
        end

        % Calculate eigenvectors
        [eigVectors, eigValues] = eig(S_B, S_W);
        [~, order] = sort(diag(eigValues), 'descend');
        eigVectors = eigVectors(:, order); % Arrange eigenvectors by eigenvalue magnitude
        
        % Projecting Data
        % Choose the number of discriminants, which is min(number of classes-1, features)
        % dimLDA = min(numDirections-1, 10);
        dimLDA = 6;
        eigVectors = eigVectors(:, 1:dimLDA);
        componentsLDA = componentsPCA * eigVectors; % Projected data
        weights = componentsLDA' * (intervalArray - mean(intervalArray, 1));
        
        % Store relevant weights for KNN
        modelParameters.classify(intervalIndex).wLDA_kNN = weights;
        modelParameters.classify(intervalIndex).dPCA_kNN = dimPCA;
        modelParameters.classify(intervalIndex).dLDA_kNN = dimLDA;
        modelParameters.classify(intervalIndex).wOpt_kNN = componentsLDA;
        modelParameters.classify(intervalIndex).mFire_kNN = mean(intervalArray, 1);
        

    % 5.4 Principal Component Regression
        for angle = 1: numDirections
            % get the spikes data for each direction
            xDirection = squeeze(xTest(:, :, angle));
            yDirection = squeeze(yTest(:, :, angle));

            % mean removal for PCR
            xPCR = xDirection(:, intervalIndex) - mean(xDirection(:, intervalIndex)); % taking mean of x hand position at the end of the interval
            yPCR = yDirection(:, intervalIndex) - mean(yDirection(:, intervalIndex)); % taking mean of y hand position at the end of the interval
            disp(size(xPCR));

            % 6.1 PCA
            % select the firing data that corresponds to the iteratively increasing intervals and the given angle
            intervalAngleArray = intervalArray(labels == angle, :);
            [components , eigenvalues] = calcPCA(intervalAngleArray);

            % use variance explained to select how many components from PCA
            explained = sort(eigenvalues/sum(eigenvalues), 'descend'); % sort in descending order, variance explained
            cumExplained = cumsum(explained);
            dimPCA = find(cumExplained >= PCAThreshold, 1, 'first'); % threshold for selecting components is 80% variance
            components = components(:, end-(dimPCA)+1:end); % components are in the order of ascending order so select from end
            disp(size(components));

            % project windowed data onto the selected components 
            projection = components' * (intervalAngleArray - mean(intervalAngleArray, 2))';
            disp(size(projection));

            % calculate regression coefficients
            xCoeff = (components * inv(projection*projection') * projection) * xPCR;
            yCoeff = (components * inv(projection*projection') * projection) * yPCR;

            % record model parameters
            modelParameters.pcr(angle, intervalIndex).xM = xCoeff;
            modelParameters.pcr(angle, intervalIndex).yM = yCoeff;
            modelParameters.pcr(angle, intervalIndex).fMean = mean(intervalAngleArray, 1);
            modelParameters.averages(intervalIndex).xMean = squeeze(mean(xn, 1));
            modelParameters.averages(intervalIndex).yMean = squeeze(mean(yn, 1));
        end

        intervalIndex = intervalIndex + 1;
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
        %   components: data projected onto the eigenvectos, sorted in ASCENDING order
        %   eigenvalues: sorted in ASCENDING order
    %----------------------------------------------------------------------

        % Compute the eigenvectors and eigenvaues
        data0Mean = data - mean(data, 1); % mean removal across trials (features) 
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

