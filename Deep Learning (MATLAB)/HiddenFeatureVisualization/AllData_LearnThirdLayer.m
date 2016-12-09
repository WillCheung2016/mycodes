% This script performas training of the third (bottleneck) layer of a 3 
% layer autoencoder, which will be trained layer-wisedly. The training data 
% of this layer are the output of the first layer.

clear

load('ZCA5-9_soudata.mat')

inputSize3  = 50;
hiddenSize3 = 10;
sparsityParam = 0.1; 
                     
lambda = 1e-3;       
beta = 7;            
maxIter = 500;

theta = initializeParameters(hiddenSize3, inputSize3);


addpath minFunc/
options.Method = 'lbfgs'; 
options.maxIter = maxIter;	  
options.display = 'on';

[opttheta3, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize3, hiddenSize3, ...
                                   lambda, sparsityParam, ...
                                   beta, soudata), ...
                              theta, options);
                          
save('ZCA5-9_opttheta3.mat','opttheta3')

toudata = feedForwardAutoencoder(opttheta3, hiddenSize3, inputSize3, soudata);

save('ZCA5-9_toudata.mat','toudata')

fprintf('Saved\n');

