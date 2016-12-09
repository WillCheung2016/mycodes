function [W1,b1] = extractEncodingMatrix(theta, hiddenSize, visibleSize)
% This function extracts encoding weight matrix and its bias from parameter 
% vector theta.
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

end