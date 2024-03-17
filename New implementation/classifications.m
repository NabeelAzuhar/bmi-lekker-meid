%% Naive Bayes Classification

function [accuracy, classifyParams] = naive_bayes_classification(pcaArray)
    % Split data into training and testing sets
    train_ratio = 0.7;
    num_train_samples = round(size(pcaArray, 2) * train_ratio);
    train_indices = randperm(size(pcaArray, 2), num_train_samples);
    test_indices = setdiff(1:size(pcaArray, 2), train_indices);

    train_data = pcaArray(:, train_indices);
    test_data = pcaArray(:, test_indices);

    % Define labels for reaching angles
    labels = repmat(1:8, 1, size(train_data, 2) / 8);

    % Train Naive Bayes classifier
    dimPCA = size(pcaArray, 1);
    means = zeros(dimPCA, 8);
    variances = zeros(dimPCA, 8);
    for angle = 1:8
        means(:, angle) = mean(train_data(:, labels == angle), 2);
        variances(:, angle) = var(train_data(:, labels == angle), 0, 2);
    end

    % Store classification parameters in a struct
    classifyParams.means = means;
    classifyParams.variances = variances;

    % Predict class labels for testing set
    predicted_labels = zeros(1, length(test_indices));
    for i = 1:length(test_indices)
        likelihoods = zeros(8, 1);
        for angle = 1:8
            likelihoods(angle) = prod(normpdf(test_data(:, i), means(:, angle), sqrt(variances(:, angle))));
        end
        posterior_probs = likelihoods / sum(likelihoods);
        [~, predicted_labels(i)] = max(posterior_probs);
    end

    % Evaluate performance
    true_labels = repmat(1:8, 1, size(test_data, 2) / 8);
    accuracy = sum(predicted_labels == true_labels(test_indices)) / length(test_indices);
end

