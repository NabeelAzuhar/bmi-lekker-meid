% Last edit: 19/03/22
% Authors: Nabeel Azuhar Mohammed, Gloria Sun, Ioana Lazar, Alexia Badea
function [modelParameters] = positionEstimatorTrainingReport(trainingData)
    
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
    [numTrials, numDirections] = size(trainingData);
    binSize = 20; % binning resolution (ms)
    window = 30; % smoothing gaussian window std (ms)
    startTime = 320; % start time of testing (ms)
    endTime = 560; % smallest time length in training data rounded to time bin of 20ms
    nHistBins = 10; % number of past feature bins used to predict current position
    numPCAComponents = 10;% PCA components to keep
    numLDAComponents = 6;

    minTrialLen = endTime/binSize; % get idx of the last bin to be stored from each trial


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


% 3. Training to extract parameters for classification
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
        
%        % PCA and LDA plot
%         if interval == 28
%             [coeff, ~, ~, ~, explained] = pca(firingCurrent');
%             cumulative_explained = cumsum(explained);
%             desired_explained_variance = 70; % Set desired explained variance percentage
%             num_components = find(cumulative_explained >= desired_explained_variance, 1);
%             firingCurrent_pca = firingCurrent' * coeff(:, 1:num_components);
%             disp(size(firingCurrent_pca))
%             
%             num_subsets = 8;
%             subset_size = numTrials;
%             
%             % PCA plot
%             fig = figure;
%             subplot(1,2,1);
%             hold on;
%             for i = 1:num_subsets
%                 subset_start = (i - 1) * subset_size + 1;
%                 subset_end = i * subset_size;
%                 plot(firingCurrent_pca(subset_start:subset_end, 1), firingCurrent_pca(subset_start:subset_end, 2), '.', 'MarkerSize', 10);
%             end
%             hold off;
%             xlabel('PCA 1'); 
%             ylabel('PCA 2');
%             title('PCA Plot of Neuronal Activity');
% 
%             % LDA plot
%             subplot(1,2,2);
%             hold on;
%             LDAData = trainProjected';
%             for i = 1:num_subsets
%                 subset_start = (i - 1) * subset_size + 1;
%                 subset_end = i * subset_size;
%                 plot(LDAData(subset_start:subset_end, 1), LDAData(subset_start:subset_end, 2), '.', 'MarkerSize', 10);
%             end
%             hold off;
%             xlabel('LDA 1'); 
%             ylabel('LDA 2');
%             title('LDA Plot of Neuronal Activity');
% 
%             fig = gcf;
%             fig.Position(4) = fig.Position(4) + 50;
% 
%             legend('Angle 1', 'Angle 2', 'Angle 3', 'Angle 4', 'Angle 5', 'Angle 6', 'Angle 7', 'Angle 8', 'Location', 'southoutside', 'Orientation', 'horizontal');
%         end

        %%% CLASSIFICATIONS %%%
        % Classification = 'KNN', 'LinearSVM', 'KernelSVM', 'NaiveBayes'
        [modelParameters] = train_classification('KNN', modelParameters, trainProjected, testProjection, overallMean, numTrials);
        %%%
        disp(modelParameters)

        intervalIdx = intervalIdx + 1; % record the current bin index (13th bin is the 1st)
    end % end of current interval

%%%%%%%%%%%%%% PCR %%%%%%%%%%%%%%% 
    % 4. Training for PCR paparameters
   
    % 4.1. Concatenate all trials to form 
    % firingData: rows = 8 reaching angles x numTrials trials x 28 timebins, columns = 98 Neurons
    % posData: rows = 8 reaching angles x numTrials trials x 28 timebins, columns = 2 positions (x,y)
        
    firingData =[];
    posData =[];
    for direction = 1:numDirections
        for trialNum = 1:numTrials
            firingData = [firingData; dataProcessed(trialNum, direction).rates(:,1:minTrialLen)'];
            posData = [posData; dataProcessed(trialNum, direction).handPos(:,1:minTrialLen)'-dataProcessed(trialNum, direction).handPos(:,1)']; % store hand positions wrt initial position
            % posData = [posData; dataProcessed(trialNum, direction).handPos(:,1:minTrialLen)'];
        end
    end
    
    % 4.2 remove low firing neurons
    firingData(:,lowFiringNeurons) = [];
    


   
    % 4.3 Apply PCA on firingData, reducing the neurons dimension to numPCAComponents and store the tranformation to then project the test Data 

    [pcaData, pcaTransformations] = calcPCA1(firingData); %% calcPCA1 - updated PCA function - for classification it was left as it was because it was giving better results

    % only keep numComponents PCA components
    pcaData = pcaData(:,end-numPCAComponents+1:end);
    pcaTransformations = pcaTransformations(:,end-numPCAComponents+1:end);
    modelParameters.pcaTransform = pcaTransformations;



    % 4.4 Add history bins for each timebin have an array with the PCA data
    % corresponding to that timebin and nHistBins before that in pcaDataHist
    
   
    pcaDataHist = zeros(size(pcaData,1),numPCAComponents*(nHistBins+1));
    for timebin = nHistBins+1:size(pcaData,1)
        pcaDataHist(timebin,:) =  reshape(pcaData(timebin-nHistBins:timebin,:)', 1, []);
    end 

    % remove first nHistBins from each trial

    rowsToDelete = [];
    for trialNum=1:numTrials*numDirections
        rowsToDelete = cat(1, rowsToDelete, (1 + (trialNum - 1) * minTrialLen : (trialNum - 1) * minTrialLen + nHistBins)');
    end

    pcaDataHist(rowsToDelete, :) = [];
    posData(rowsToDelete, :) = [];

  
    % 4.5. Principal Component Regression 
    % xn, yn = x, y position data padded with the last position value of each trial
    % x, y = SAMPLED x, y position data using the set bin size (20ms)

    R2_table = zeros(numDirections, 2);
    
    for direction = 1:numDirections

      pcaDataPerDir = pcaDataHist(numTrials*(minTrialLen-nHistBins)*(direction-1)+1:numTrials*(minTrialLen-nHistBins)*direction,:);
      posDataPerDir_x = posData(numTrials*(minTrialLen-nHistBins)*(direction-1)+1:numTrials*(minTrialLen-nHistBins)*direction,1);
      posDataPerDir_y = posData(numTrials*(minTrialLen-nHistBins)*(direction-1)+1:numTrials*(minTrialLen-nHistBins)*direction,2);
      modelParameters.averages(direction).xMean = squeeze(mean(posDataPerDir_x)); % squeeze(mean(xPadded, 1)) = (792ms x 8), mean across 100 trials for each angle
      modelParameters.averages(direction).yMean = squeeze(mean(posDataPerDir_y));
      posDataPerDir_x = posDataPerDir_x - mean(posDataPerDir_x);
      posDataPerDir_y = posDataPerDir_y - mean(posDataPerDir_y);

      xCoeff = inv(pcaDataPerDir' *pcaDataPerDir) * pcaDataPerDir' * posDataPerDir_x; 
      yCoeff = inv(pcaDataPerDir'*pcaDataPerDir) * pcaDataPerDir' * posDataPerDir_y;

      % calculate R2
        
        predicted_x = pcaDataPerDir * xCoeff;
        predicted_y = pcaDataPerDir * yCoeff;
        
        % Total Sum of Squares (SST)
        mean_x = mean(posDataPerDir_x);
        mean_y = mean(posDataPerDir_y);
        SST_x = sum((posDataPerDir_x - mean_x).^2);
        SST_y = sum((posDataPerDir_y - mean_y).^2);
        
        % Residual Sum of Squares (SSE)
        SSE_x = sum((posDataPerDir_x - predicted_x).^2);
        SSE_y = sum((posDataPerDir_y - predicted_y).^2);
        
        %  R^2
        R2_x = 1 - (SSE_x / SST_x);
        R2_y = 1 - (SSE_y / SST_y);
    
        R2_table(direction, 1) = R2_x;
        R2_table(direction, 2) = R2_y;


      modelParameters.pcr(direction).xM = xCoeff;
      modelParameters.pcr(direction).yM = yCoeff;

    end

    % Display the R2 table with column and row headers
    disp(' ')
    disp('Direction   R2_x   R2_y');
    disp(['           ', '----------------------']);
    for direction = 1:numDirections
        disp([sprintf('Direction %d |', direction), sprintf(' %.4f | %.4f', R2_table(direction, :))]);
    end
   
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
                    % handPosBins(:, bin)= handPosData(1:2, binIndices(bin));
                end
                spikeBins = sqrt(spikeBins);
                % spikeBins = log(1+spikeBins);
                % spikeBins = 2*sqrt(spikeBins + 3/8);

                % fill up the output
                dataProcessed(trial, angle).spikes = spikeBins; % spikes are now binned
                dataProcessed(trial, angle).handPos = handPosBins; % select only x and y
            end
        end
    
        % Gaussian Smoothing
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

        % % Rectangular Smoothing
        % % Generating the Rectangular window
        % windowWidth = 10 * (window/binSize); % define the width of the rectangular window
        % rectWindow = ones(1, windowWidth) / windowWidth; % create a rectangular window and normalize it
        % 
        % % add smoothed firing rates to the processed data
        % for angle = 1 : size(dataProcessed, 2)
        %     for trial = 1 : size(dataProcessed, 1)
        %         firingRates = zeros(size(dataProcessed(trial, angle).spikes, 1), size(dataProcessed(trial, angle).spikes, 2));
        %         % convolve window with each neuron for smoothing
        %         for neuron = 1 : size(dataProcessed(trial, angle).spikes, 1)
        %             firingRates(neuron, :) = conv(dataProcessed(trial, angle).spikes(neuron, :), rectWindow, 'same') / (binSize/1000);
        %         end
        %         dataProcessed(trial, angle).rates = firingRates; % add rates as a new field to processed data
        %     end
        % end

        % % Triangular smoothing
        % % Define the width of the triangular window
        % windowWidth = 10 * (window/binSize); % Adjust 'window' and 'binSize' as per your setup
        % % Create a triangular window
        % triWindow = triang(windowWidth);
        % % Normalize the window to ensure its sum equals one
        % triWindow = triWindow / sum(triWindow);
        % 
        % % Loop through trials and angles
        % for angle = 1 : size(dataProcessed, 2)
        %     for trial = 1 : size(dataProcessed, 1)
        %         % Preallocate firingRates array for storing smoothed data
        %         firingRates = zeros(size(dataProcessed(trial, angle).spikes, 1), size(dataProcessed(trial, angle).spikes, 2));
        % 
        %         % Convolve the spike data with the triangular window for each neuron
        %         for neuron = 1 : size(dataProcessed(trial, angle).spikes, 1)
        %             % Perform convolution and normalize by the bin size converted to seconds
        %             firingRates(neuron, :) = conv(dataProcessed(trial, angle).spikes(neuron, :), triWindow, 'same') / (binSize/1000);
        %         end
        % 
        %         % Store the computed firing rates back into the processed data structure
        %         dataProcessed(trial, angle).rates = firingRates;
        %     end
        % end

    end % end of function dataProcessor


    function [dataTransformed, eigenvectors] = calcPCA1(data)
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

        dataTransformed = data * eigenvectors;
    
    end % end of calcPCA function
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

    function [modelParameters] = saveLinearSVM(modelParameters, trainProjected, testProjection, overallMean, numTrials)
        % Linear SVM Classification
        directions = [1,2,3,4,5,6,7,8];
        y_labels = repelem(directions, numTrials)';
        svm_template = templateSVM('KernelFunction', 'linear');
        ecoc_model = fitcecoc(trainProjected', y_labels, 'Learners', svm_template);
        
       % Store all the relevant weights for KNN
        modelParameters.LinearSVMClassify(intervalIdx).ecoc_model = ecoc_model;
        modelParameters.LinearSVMClassify(intervalIdx).testProjection = testProjection; % (2660 x 6)
        modelParameters.LinearSVMClassify(intervalIdx).meanFiring = overallMean; % (2660 x 1), mean rate for each neuron-bin
    end

    
    function [modelParameters] = saveKernelSVM(modelParameters, trainProjected, testProjection, overallMean, numTrials)
        % Linear SVM Classification
        directions = [1,2,3,4,5,6,7,8];
        y_labels = repelem(directions, numTrials)';
        svm_template = templateSVM('KernelFunction', 'rbf','BoxConstraint', 100, 'KernelScale', 100);
        ecoc_model = fitcecoc(trainProjected', y_labels, 'Learners', svm_template);
        
       % Store all the relevant weights for KNN
        modelParameters.KernelSVMClassify(intervalIdx).ecoc_model = ecoc_model;
        modelParameters.KernelSVMClassify(intervalIdx).testProjection = testProjection; % (2660 x 6)
        modelParameters.KernelSVMClassify(intervalIdx).meanFiring = overallMean; % (2660 x 1), mean rate for each neuron-bin
    end


    function [modelParameters] = saveNaiveBayes(modelParameters, trainProjected, testProjection, overallMean, numTrials)
        % Linear SVM Classification
        directions = [1,2,3,4,5,6,7,8];
        y_labels = repelem(directions, numTrials)';
        nb_classifier = fitcnb(trainProjected', y_labels);
        
       % Store all the relevant weights for KNN
        modelParameters.NaiveBayesClassify(intervalIdx).nb_classifier = nb_classifier;
        modelParameters.NaiveBayesClassify(intervalIdx).testProjection = testProjection; % (2660 x 6)
        modelParameters.NaiveBayesClassify(intervalIdx).meanFiring = overallMean; % (2660 x 1), mean rate for each neuron-bin
    end


    function [modelParameters] = train_classification(classification, modelParameters, trainProjected, testProjection, overallMean, numTrials)
        if strcmp(classification, 'KNN')
            [modelParameters] = saveKNN(modelParameters, trainProjected, testProjection, overallMean);
        elseif strcmp(classification, 'LinearSVM')
            [modelParameters] = saveLinearSVM(modelParameters, trainProjected, testProjection, overallMean, numTrials);
        elseif strcmp(classification, 'KernelSVM')
            [modelParameters] = saveKernelSVM(modelParameters, trainProjected, testProjection, overallMean, numTrials);
        elseif strcmp(classification, 'NaiveBayes')
            [modelParameters] = saveNaiveBayes(modelParameters, trainProjected, testProjection, overallMean, numTrials);
        end
    end
end
