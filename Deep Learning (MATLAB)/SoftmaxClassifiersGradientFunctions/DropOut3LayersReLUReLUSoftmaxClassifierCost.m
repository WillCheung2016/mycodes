function [cost,grad] = ...
DropOut3LayersReLUReLUSoftmaxClassifierCost(theta, inputSize, hiddenSize1, ...
                     hiddenSize2,outputSize,data, labels_vec, Mask1, Mask2)
% This function computes the loss value and the gradient given theta. Mask1
% and Mask2 are expected to be generated by sampling from Bernoulli
% distribution.
% Input Arguments:
%                theta - vector of model parameters
%                inputSize, hiddenSize1, hiddenSize2, outputSize -
%                           architecture parameters
%                data - matrix of input data
%                labels_vec - matrix of training labels
%                Mask1 - dropout mask for the first hidden layer
%                Mask2 - dropout mask for the second hidden layer
% Output Arguments:
%                cost - the value of loss function for the given theta
%                grad - the gradient given theta

theta1 = theta(1:inputSize*hiddenSize1+hiddenSize1);
theta2 = theta(inputSize*hiddenSize1+hiddenSize1+1:inputSize*hiddenSize1+hiddenSize1+hiddenSize1*hiddenSize2+hiddenSize2);
theta3 = theta(inputSize*hiddenSize1+hiddenSize1+hiddenSize1*hiddenSize2+hiddenSize2+1:end);
[W1,b1] = extractWeights(theta1, inputSize, hiddenSize1);
[W2,b2] = extractWeights(theta2, hiddenSize1, hiddenSize2);
[W3,b3] = extractWeights(theta3, hiddenSize2, outputSize);
N = size(data,2);
display_network(W1');

Z1 = W1*data+repmat(b1,[1,N]);
Extracted_features = ReLU(Z1);
Extracted_features = Extracted_features .* Mask1;
Z2 = W2*Extracted_features+repmat(b2,[1,N]);
Hidden_output = ReLU(Z2);
Hidden_output = Hidden_output .* Mask2;

groundTruth = full(sparse(labels_vec, 1:N, 1, outputSize, N));

Powers = W3 * Hidden_output + repmat(b3,[1,N]);
alpha = max(Powers,[],1);
Powers = bsxfun(@minus,Powers,alpha) ;
Exp_Mat = exp(Powers);
Sum_Vec = sum(Exp_Mat);
log_sum = 0;
for i = 1:N
    j = labels_vec(i);
    log_sum = log_sum + log(Exp_Mat(j,i)/Sum_Vec(i));
end
cost = (-1/N)*log_sum;

Del_nl= Exp_Mat./repmat(Sum_Vec,[outputSize,1]) - groundTruth;
W3grad = (1/N) * Del_nl * Hidden_output';
b3grad = (1/N) * sum(Del_nl,2);

dReLU_activation1 = difReLU(Z1);
dReLU_activation1 = dReLU_activation1 .* Mask1;
dReLU_activation2 = difReLU(Z2);
dReLU_activation2 = dReLU_activation2 .* Mask2;

Del_3 = (W3'*Del_nl).*dReLU_activation2;
W2grad = (1/N)*Del_3 * Extracted_features';
b2grad = (1/N)*sum(Del_3,2);

Del_2 = (W2'*Del_3).*dReLU_activation1;
W1grad = (1/N)*Del_2 * data';
b1grad = (1/N)*sum(Del_2,2);

%disp([T1 T2 S])

grad = [W1grad(:) ; b1grad(:) ; W2grad(:) ; b2grad(:) ; W3grad(:) ; b3grad(:)];

end

