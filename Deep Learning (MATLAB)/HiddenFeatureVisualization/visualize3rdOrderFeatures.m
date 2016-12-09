% This script reads weight matrices of a trained 3-layers autoencoder using
% MNIST data of digits from 5 to 9, and visualize the learned features in
% the third layer (the bottlenect layer) by gradient ascend or constraint 
% least square method.

clear

load('ZCA5-9_opttheta.mat')
load('ZCA5-9_opttheta2.mat')
load('ZCA5-9_opttheta3.mat')

inputSize1  = 28 * 28;
hiddenSize1 = 200;

inputSize2  = 200;
hiddenSize2 = 50;

inputSize3  = 50;
hiddenSize3 = 10;

[W1,b1] = extractEncodingMatrix(opttheta, hiddenSize1, inputSize1);
[W2,b2] = extractEncodingMatrix(opttheta2, hiddenSize2, inputSize2);
[W3,b3] = extractEncodingMatrix(opttheta3, hiddenSize3, inputSize3);

SecLF = [];
act_vals = [];
for i =1:hiddenSize3
    w = W3(i,:)';
    b = b3(i);
    [vector, values]= findGoodInput3_ContLeastSq(w,b,W1,b1,W2,b2,5e-2);
    SecLF = [SecLF vector];
    act_vals = [act_vals values];
end
display_network(SecLF);