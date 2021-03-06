function [cost,grad] = DropoutAutoencoderTiedWeightsReLUSigKLCost(theta, visibleSize, hiddenSize, ...
                                             data, cor_data, Mask)
% This function computes the gradient and loss of the autoencoder model 
% given current model parameter vector theata, data, and data corrupted by
% random noise.
% Details: Cross-entropy Loss
%          ReLU Activation Units
%          Tied Weights
W = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(hiddenSize*visibleSize+1:hiddenSize*visibleSize+hiddenSize);
b2 = theta(hiddenSize*visibleSize+hiddenSize+1:end);

W1 = W;
W2 = W';

display_network(W1');

Z1 = W1*cor_data+repmat(b1,[1,size(data,2)]);
Hidden_output = ReLU(Z1);
Hidden_output = Hidden_output .* Mask;
Z2 = W2*Hidden_output+repmat(b2,[1,size(data,2)]);
Output=sigmoid(Z2);

Sigmoid_activation = difReLU(Z1);
Sigmoid_activation = Sigmoid_activation .* Mask;

Del_nl=((1-data)./(1-Output) - data./Output).*(Output.*(1-Output));
W2grad = (1/size(Del_nl,2))*Del_nl * Hidden_output';
b2grad = (1/size(Del_nl,2))*sum(Del_nl,2);

Del_2 = (W2'*Del_nl).*Sigmoid_activation;
W1grad = (1/size(data,2))*Del_2 * cor_data';
b1grad = (1/size(data,2))*sum(Del_2,2);

Wgrad = W1grad + W2grad';

cost = (-1/size(data,2))*sum(sum(data.*log(Output)+(1-data).*log(1-Output)));

%disp([T1 T2 S])

grad = [Wgrad(:) ; b1grad(:) ; b2grad(:)];

end