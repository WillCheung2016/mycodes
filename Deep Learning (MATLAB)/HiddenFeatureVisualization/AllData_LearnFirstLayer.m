% This script performas training of the first layer of a 3 layer
% autoencoder, which will be trained layer-wisedly. The training data for
% this layer are the whitened raw data.

clear
inputSize  = 28 * 28;
hiddenSize = 200;
sparsityParam = 0.1; 
                    
lambda = 3e-3;       
beta = 3;            
maxIter = 400;

load('selected_mnist.mat')

theta = initializeParameters(hiddenSize, inputSize);
[mnistData, ZCAW] =ZCAwhiten(selected_mnist,0.01);

addpath minFunc/
options.Method = 'lbfgs'; 
                          
options.maxIter = maxIter;	  
options.display = 'on';

[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, selected_mnist), ...
                              theta, options);

                          
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network((W1*ZCAW)');
save('ZCA5-9_opttheta.mat','opttheta','ZCAW')
foudata = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, selected_mnist);
save('ZCA5-9all_foudata.mat','foudata')
fprintf('Saved\n');

% trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, trainData);
% testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, testData);
% options.maxIter = 100;
% lambda = 1e-4;
% softmaxModel = softmaxTrain(hiddenSize, numLabels, lambda, trainFeatures, trainLabels, options);
% [pred] = softmaxPredict(softmaxModel, testFeatures);
% fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));
