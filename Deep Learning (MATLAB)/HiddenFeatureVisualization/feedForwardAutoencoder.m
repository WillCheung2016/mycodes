% This function computes the activation of the first layer of an
% autoencoder with sigmoid activation function.

function [activation] = feedForwardAutoencoder(theta, hiddenSize, visibleSize, data)

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
activation = sigmoid(W1 * data + repmat(b1,[1,size(data,2)]));

end