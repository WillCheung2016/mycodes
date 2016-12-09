% This script performas training of the second layer of a 3 layer
% autoencoder, which will be trained layer-wisedly. The training data of
% this layer are the output of the first layer.

clear

load('ZCA5-9_opttheta.mat')
load('ZCA5-9all_foudata.mat')
inputSize2  = 200;
hiddenSize2 = 50;
sparsityParam = 0.25; 
                     
lambda = 3e-3;       
beta = 3;            
maxIter = 400;



theta = initializeParameters(hiddenSize2, inputSize2);


addpath minFunc/
options.Method = 'lbfgs'; 
options.maxIter = maxIter;	  
options.display = 'on';

[opttheta2, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize2, hiddenSize2, ...
                                   lambda, sparsityParam, ...
                                   beta, foudata), ...
                              theta, options);
                          
save('ZCA5-9_opttheta2.mat','opttheta2')

soudata = feedForwardAutoencoder(opttheta2, hiddenSize2, inputSize2, foudata);
save('ZCA5-9_soudata.mat','soudata')

fprintf('Saved\n');

