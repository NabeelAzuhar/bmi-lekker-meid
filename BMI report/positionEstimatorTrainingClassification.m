function [modelParameters] = positionEstimatorTrainingClassification(trainingData, trial_split)

%--------------------------------------------------------------------------
    % Arguments:
    
    % - training_data:
    %     training_data(n,k)              (n = trial id,  k = reaching angle)
    %     training_data(n,k).trialId      unique number of the trial
    %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
    %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
    
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


% 2. Generate firing data matrix + handPos data matrix
    % initialisations
    firingData = zeros(numNeurons*endTime/binSize, numDirections*numTrials); % firing data (binned, truncated to 28 bins)
    handPosData = zeros(2*endTime/binSize, numTrials*numDirections); % handPos data (binned, truncated to 28 bins)
    xPadded = zeros(numTrials, maxTimeSteps, numDirections); % original handPos data (in ms) but padded with the longest trajectory
    yPadded = zeros(numTrials, maxTimeSteps, numDirections);
    
    % 2.1 Generate firing and hand position data matrices
    for angle = 1 : numDirections % each angle
        for trial = 1 : numTrials % each trial
            % generate padded hand position data
            xPadded(trial, :, angle) = [trainingData(trial,angle).handPos(1,:), trainingData(trial,angle).handPos(1, end) * ones(1, maxTimeSteps - timeSteps(numTrials*(angle-1) + trial))];
            yPadded(trial, :, angle) = [trainingData(trial,angle).handPos(2,:), trainingData(trial,angle).handPos(2, end) * ones(1, maxTimeSteps - timeSteps(numTrials*(angle-1) + trial))]; 

            for bin = 1 : endTime/binSize % each time bin -> taking 28 time bins based on 560ms total time
                % generate firing data
                firingData(numNeurons*(bin-1)+1 : numNeurons*bin, numTrials*(angle-1)+trial) = dataProcessed(trial, angle).rates(:, bin);   
                % generate handPos data
                handPosData(2*(bin-1)+1 : 2*bin, numTrials*(angle-1)+trial) = dataProcessed(trial, angle).handPos(:, bin);  
            end
        end
    end
    % Out: firingData (98*28 x 8*100): rows = 98 neurons * 28 bins, columns = 8 angles * 100 trials
    %      handPosData = (28*2 x 8*100) - binned averaged handPos
    %      xPadded (or yPadded) = (100 x 792 x 8) - originial handPos data in ms padded with the last value to the longest trajectory length

    % 2.2 Mark the neurons to be removed based on firing rate (with rate threshold at 0.5)
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
    numNeurons = numNeurons - length(lowFiringNeurons); % update number of neurons after removing the low firers
    modelParameters.lowFirers = lowFiringNeurons; % record low firing neurons to remove the same ones in test data
    % Out:
    %   firingData: filtered rows after neuron removal (removes row corresponding to low firing neurons)


% 3. Training to extract parameters
    binIndices = (startTime:binSize:endTime) / binSize; % 16 : 28
    intervalIdx = 1;

    % Start with data from 320ms (16th bin), then iteratively add 20ms (1 bin) until 560ms (28th bin) for training
    for interval = binIndices % iteratively add testing time: 16, 17, 18, ... 28

    % 3.1 get firing rate data up to the certain time bin
        firingCurrent = zeros(numNeurons*interval, numDirections*numTrials);
        for bin = 1 : interval
            firingCurrent(numNeurons*(bin-1)+1:numNeurons*bin, :) = firingData(numNeurons*(bin-1)+1:numNeurons*bin, :);
        end
    % Out: firingCurrent = firing rates data up to the current specified time interval (interval*95, 8*100)

    % 3.2 Principal Component Analysis for dimensionality reduction
        [eigenvectors, eigenvalues] = calcPCA(firingCurrent);
        % Out:
        %   eigenvectors: eigenvectors in ASCENDING order
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
    
    % 3.3 Linear Discriminant Analysis (for knn classification)
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
        testProjection = pcaProjection * eigenvectors(:, sortIdx(1:dimLDA)); % LDA projection components for testing data to project on (2660 x 6)
        trainProjected = testProjection' * (firingCurrent - overallMean); % training data projected onto LDA (6x2660) * (2660x800) = (6 x 800)

        %%% CLASSIFICATIONS %%%
        % Classification = 'KNN', 'LinearSVM', 'KernelSVM', 'NaiveBayes'
        [modelParameters] = train_classification('NaiveBayes', modelParameters, trainProjected, testProjection, overallMean);
        %%%
        disp(modelParameters)

    % 4.4 Principal Component Regression
        for angle = 1: numDirections % generate coefficients for each angle
            % select data for the specified angle
            xPos = handPosData((startTime/binSize * 2 - 1):2:end, labels==angle); % select the x pos for the selected angle (28x100), starting from 13th bin
            yPos = handPosData((startTime/binSize * 2):2:end, labels==angle); % (13 x 100), 13 bins from 320ms to 560ms (1, 2, 3... 13)

            % select data for the current time bin (e.g. 13, 14 ... 28)
            xCurrent = xPos(intervalIdx, :) - mean(xPos(intervalIdx, :)); % (1x100), mean removed
            yCurrent = yPos(intervalIdx, :) - mean(yPos(intervalIdx, :));

            % select the firing data that corresponds to the current interval AND the given angle
            firingWindowed = firingCurrent(:, labels == angle); % firing data for current interval, selected angle e.g. (2660x100)

            % eigenanalysis
            [eigenvectors, eigenvalues] = calcPCA(firingWindowed);
            explained = sort(eigenvalues/sum(eigenvalues), 'descend'); % sort in descending order, variance explained
            cumExplained = cumsum(explained);
            dimPCA = find(cumExplained >= pcaThreshold, 1, 'first');
            eigenvectors = eigenvectors(:, end-(dimPCA)+1:end); % (100 x dimPCA)
            pcaProjection = firingWindowed * eigenvectors; % (2660x100) * (100xdimPCA) = (2660xdimPCA)
            projection = pcaProjection' * (firingWindowed - mean(firingWindowed, 1)); % e.g. (dimPCAx2660) * (2660x100) = (dimPCAx100)
    
            % calculate regression coefficients - used to multiply with testing data
            xCoeff = (pcaProjection * inv(projection*projection') * projection) * xCurrent'; % (2660xdimPCA * (dimPCA x dimPCA) * (dimPCAx100)) * (100x1)
            yCoeff = (pcaProjection * inv(projection*projection') * projection) * yCurrent'; % = (2660x1)
    
            % record model parameters
            modelParameters.regression(angle, intervalIdx).xCoeff = xCoeff;
            modelParameters.regression(angle, intervalIdx).yCoeff = yCoeff;
            modelParameters.regression(angle, intervalIdx).firingMean = mean(firingWindowed, 1); % (1x800)
            modelParameters.positionMeans(intervalIdx).xMean = squeeze(mean(xPadded, 1)); % squeeze(mean(xPadded, 1)) = (792ms x 8), mean across 100 trials for each angle
            modelParameters.positionMeans(intervalIdx).yMean = squeeze(mean(yPadded, 1));

        end % end of PCR

        intervalIdx = intervalIdx + 1; % record the current bin index (13th bin is the 1st)
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

    
    function [modelParameters] = saveKNN(modelParameters, trainProjected, testProjection, overallMean)
        % Store all the relevant weights for KNN
        modelParameters.knnClassify(intervalIdx).trainProjected = trainProjected; % (6 x 800) - 800 trials projected onto 6 components
        modelParameters.knnClassify(intervalIdx).testProjection = testProjection; % (2660 x 6)
        modelParameters.knnClassify(intervalIdx).meanFiring = overallMean; % (2660 x 1), mean rate for each neuron-bin
    end

    function [modelParameters] = saveLinearSVM(modelParameters, trainProjected, testProjection, overallMean)
        % Linear SVM Classification
        directions = [1,2,3,4,5,6,7,8];
        y_labels = repelem(directions, trial_split)';
        svm_template = templateSVM('KernelFunction', 'linear');
        ecoc_model = fitcecoc(trainProjected', y_labels, 'Learners', svm_template);
        
       % Store all the relevant weights for KNN
        modelParameters.LinearSVMClassify(intervalIdx).ecoc_model = ecoc_model;
        modelParameters.LinearSVMClassify(intervalIdx).testProjection = testProjection; % (2660 x 6)
        modelParameters.LinearSVMClassify(intervalIdx).meanFiring = overallMean; % (2660 x 1), mean rate for each neuron-bin
    end

    
    function [modelParameters] = saveKernelSVM(modelParameters, trainProjected, testProjection, overallMean)
        % Linear SVM Classification
        directions = [1,2,3,4,5,6,7,8];
        y_labels = repelem(directions, trial_split)';
        svm_template = templateSVM('KernelFunction', 'rbf','BoxConstraint', 100, 'KernelScale', 100);
        ecoc_model = fitcecoc(trainProjected', y_labels, 'Learners', svm_template);
        
       % Store all the relevant weights for KNN
        modelParameters.KernelSVMClassify(intervalIdx).ecoc_model = ecoc_model;
        modelParameters.KernelSVMClassify(intervalIdx).testProjection = testProjection; % (2660 x 6)
        modelParameters.KernelSVMClassify(intervalIdx).meanFiring = overallMean; % (2660 x 1), mean rate for each neuron-bin
    end


    function [modelParameters] = saveNaiveBayes(modelParameters, trainProjected, testProjection, overallMean)
        % Linear SVM Classification
        directions = [1,2,3,4,5,6,7,8];
        y_labels = repelem(directions, trial_split)';
        nb_classifier = fitcnb(trainProjected', y_labels);
        
       % Store all the relevant weights for KNN
        modelParameters.NaiveBayesClassify(intervalIdx).nb_classifier = nb_classifier;
        modelParameters.NaiveBayesClassify(intervalIdx).testProjection = testProjection; % (2660 x 6)
        modelParameters.NaiveBayesClassify(intervalIdx).meanFiring = overallMean; % (2660 x 1), mean rate for each neuron-bin
    end


    function [modelParameters] = train_classification(classification, modelParameters, trainProjected, testProjection, overallMean)
        if strcmp(classification, 'KNN')
            [modelParameters] = saveKNN(modelParameters, trainProjected, testProjection, overallMean);
        elseif strcmp(classification, 'LinearSVM')
            [modelParameters] = saveLinearSVM(modelParameters, trainProjected, testProjection, overallMean);
        elseif strcmp(classification, 'KernelSVM')
            [modelParameters] = saveKernelSVM(modelParameters, trainProjected, testProjection, overallMean);
        elseif strcmp(classification, 'NaiveBayes')
            [modelParameters] = saveNaiveBayes(modelParameters, trainProjected, testProjection, overallMean);
        end
    end

% Nested functions --------------------------------------------------------

end