function [cost,grad] = sparseAutoencoderReLUSigCostL1Cauchy(theta, visibleSize, hiddenSize, ...
                                             lambda, beta, data,cor_data)

% This function computes the gradient and loss of the autoencoder model 
% given current model parameter vector theata, data, and data corrupted by
% random noise.There are two extra terms in the loss function: one is to
% penalize activations of hidden nodes, the other is to encourage features
% leared expand in the feature space as much as possible.
% Details: Square Loss
%          ReLU Activation Units
%          Cauchy penalty on hidden nodes activations
%          Penalty on Eigenvalues of learned features

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

alpha = 1e10;
Z1 = W1*cor_data+repmat(b1,[1,size(data,2)]);
Hidden_output = ReLU(Z1);
Z2 = W2*Hidden_output+repmat(b2,[1,size(data,2)]);
Output=sigmoid(Z2);

Sigmoid_activation = difReLU(Z1);
S = sum(sum(log(1+Hidden_output.^2)));

Del_nl=(Output-data).*(Output.*(1-Output));

Del_2 = (W2'*Del_nl).*Sigmoid_activation;
% +beta*repmat(sparsity_vec,[1,size(data,2)])

SA1 = softAbs(W1,alpha);
dSA1 = dSoftAbs(W1,alpha);
SA2 = softAbs(W2,alpha);
dSA2 = dSoftAbs(W2,alpha);

sparsity_grad = (2*(Hidden_output.*Sigmoid_activation)./(1+Hidden_output.^2));

W1grad = (1/size(data,2))*Del_2 * cor_data' + lambda*dSA1 + beta*sparsity_grad*cor_data';
b1grad = (1/size(data,2))*sum(Del_2,2) + beta*sum(sparsity_grad,2);

W2grad = (1/size(Del_nl,2))*Del_nl * Hidden_output' + lambda*dSA2;
b2grad = (1/size(Del_nl,2))*sum(Del_nl,2);

T1 = (norm(Output-data,'fro'))^2;
T2 = sum(sum(SA1))+sum(sum(SA2));
cost = (1/(2*size(data,2)))* T1+ lambda*T2 + beta*S;

%disp([T1 T2 S])

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end
