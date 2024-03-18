%% Naive Bayes Classification

function [classifyParams] = naiveBayesClassification(pcaArray)
    %----------------------------------------------------------------------
    % Performs the naive bayes classification and returns the 
    % classification parameters
    %     
    % Arguments:
        %   pcaArray: flattened array obtained after PCA

        % Return Value:
        %   classifyParams: structure containing the mean and variance for
        %   prediction
    %----------------------------------------------------------------------
    % Define labels for reaching angles
    labels = repmat(1:8, 1, size(pcaArray, 2) / 8);

    % Train Naive Bayes classifier
    dimPCA = size(pcaArray, 1);
    means = zeros(dimPCA, 8);
    variances = zeros(dimPCA, 8);
    for angle = 1:8
        means(:, angle) = mean(pcaArray(:, labels == angle), 2);
        variances(:, angle) = var(pcaArray(:, labels == angle), 0, 2);
    end

    % Store classification parameters in a struct
    classifyParams.means = means;
    classifyParams.variances = variances;
end

function predictedLabels = naiveBayesClassificationPred(array, means, variances)
    %----------------------------------------------------------------------
    % Predicts the direction of the new data using the classification data
    % from naive bayes classification
    %     
    % Arguments:
        %   array: flattened array describing the new data 
        %   means: the means from naive bayes classification 
        %   variances: the variances from naive bayes classification

        % Return Value:
        %   predictedLabels: predicted direction(s) of new data
    %----------------------------------------------------------------------
    % Predict class labels for testing set
    predictedLabels = zeros(1, size(array, 2));
    for i = 1:size(array, 2)
        likelihoods = zeros(8, 1);
        for angle = 1:8
            likelihoods(angle) = prod(normpdf(array(:, i), means(:, angle), sqrt(variances(:, angle))));
        end
        posteriorProbs = likelihoods / sum(likelihoods);
        [~, predictedLabels(i)] = max(posteriorProbs);
    end
end

function [accuracy, classifyParams] = naiveBayesClassificationCheck(pcaArray)
    %----------------------------------------------------------------------
    % Performs the naive bayes classification and returns the 
    % classification parameters and checks the accuracy of the
    % classification by training it on 70% of the array and testing it on
    % the remaining 30%
    %     
    % Arguments:
        %   pcaArray: flattened array obtained after PCA

        % Return Value:
        %   classifyParams: structure containing the mean and variance for
        %   prediction 
        %   accuracy: float measuring the accuracy of the classsification
    %----------------------------------------------------------------------

    % Split data into training and testing sets
    trainRatio = 0.7;
    numTrainSamples = round(size(pcaArray, 2) * trainRatio);
    trainIndices = randperm(size(pcaArray, 2), numTrainSamples);
    testIndices = setdiff(1:size(pcaArray, 2), trainIndices);

    trainData = pcaArray(:, trainIndices);
    testData = pcaArray(:, testIndices);

    % Define labels for reaching angles
    labels = repmat(1:8, 1, size(trainData, 2) / 8);

    % Train Naive Bayes classifier
    dimPCA = size(pcaArray, 1);
    means = zeros(dimPCA, 8);
    variances = zeros(dimPCA, 8);
    for angle = 1:8
        means(:, angle) = mean(trainData(:, labels == angle), 2);
        variances(:, angle) = var(trainData(:, labels == angle), 0, 2);
    end

    % Store classification parameters in a struct
    classifyParams.means = means;
    classifyParams.variances = variances;

    % Predict class labels for testing set
    predictedLabels = zeros(1, length(testIndices));
    for i = 1:length(testIndices)
        likelihoods = zeros(8, 1);
        for angle = 1:8
            likelihoods(angle) = prod(normpdf(testData(:, i), means(:, angle), sqrt(variances(:, angle))));
        end
        posteriorProbs = likelihoods / sum(likelihoods);
        [~, predictedLabels(i)] = max(posteriorProbs);
    end

    % Evaluate performance
    trueLabels = repmat(1:8, 1, size(testData, 2) / 8);
    accuracy = sum(predictedLabels == trueLabels(testIndices)) / length(testIndices);
end
