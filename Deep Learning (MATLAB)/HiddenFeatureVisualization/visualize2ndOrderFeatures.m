% This script reads weight matrices of the first and the second layers of
% a trained autoencoder using MNIST data of digits from 5 to 9, and
% visualize the learned features in the second layer by gradient ascend or
% constraint least square method.

clear

load('ZCA5-9_opttheta.mat')
load('ZCA5-9_opttheta2.mat')

inputSize1  = 28 * 28;
hiddenSize1 = 200;

inputSize2  = 200;
hiddenSize2 = 50;

[W1,b1] = extractEncodingMatrix(opttheta, hiddenSize1, inputSize1);
[W2,b2] = extractEncodingMatrix(opttheta2, hiddenSize2, inputSize2);

SecLF = [];
act_vals = [];
for i =1:hiddenSize2
    w = W2(i,:)';
    b = b2(i);
    [vector, values]= findGoodInput2(w,b,W1,b1,5e-2);
    % findGoodInput2_ContLeastSq(w,b,W1,b1,5e-2)
    SecLF = [SecLF vector];
    act_vals = [act_vals values];
end
display_network(SecLF);