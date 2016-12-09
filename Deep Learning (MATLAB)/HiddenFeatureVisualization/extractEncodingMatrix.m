function [W1,b1] = extractEncodingMatrix(theta, hiddenSize, visibleSize)

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

end