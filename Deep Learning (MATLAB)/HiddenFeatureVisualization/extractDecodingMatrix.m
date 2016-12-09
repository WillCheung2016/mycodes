function [W2,b2] = extractDecodingMatrix(theta, hiddenSize, visibleSize)

W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

end