% Last edit: 19/03/22
% Authors: Nabeel Azuhar Mohammed, Gloria Sun, Ioana Lazar, Alexia Badea

function [x, y, modelParameters]= estimatorTest(testData, modelParameters)

    % ---------------------------------------------------------------------
    % Inputs:
        % testData: struct with very similar formatting to trial, other than it has the additinal field of starting position
        % modelParameters: previously saved modelParameters from PCA-LDA analysis using the training dataset
    
    % Outputs:
        % decodedPosX: predicted X position according to the PCR model
        % decodedPosY: predicted Y position according to the PCR model
        % newParameters: any modifications in classification etc stored here
    %----------------------------------------------------------------------
            
    % Initialisations
    binSize = 20; % binning resolution (ms)
    window = 50; % window length (ms)
    timeStart = 320; % start time of testing (ms)
    numDirections = 8;

    % Data pre-processing
    dataProcessed = dataProcessor(testData, binSize, window);
    numNeurons = size(dataProcessed(1, 1).rates, 1);
    timeTotal = size(testData.spikes, 2); % total time of the current struct data
  
    
end