function [cost,grad] = sparseAutoencoderReLUDetDIVCostL1(theta, visibleSize, hiddenSize, ...
                                             lambda1,lambda2, beta, gamma, data,cor_data)

% This function computes the gradient and loss of the autoencoder model 
% given current model parameter vector theata, data, and data corrupted by
% random noise.There are two extra terms in the loss function: one is to
% penalize activations of hidden nodes, the other is to encourage features
% leared expand in the feature space as much as possible.
% Details: Square Loss
%          ReLU Activation Units
%          L1 penalty on hidden nodes activations
%          Penalty on Eigenvalues of learned features

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

alpha = 1e10;
Z1 = W1*cor_data+repmat(b1,[1,size(data,2)]);
Hidden_output = ReLU(Z1);
Z2 = W2*Hidden_output+repmat(b2,[1,size(data,2)]);
Output=ReLU(Z2);

hiddenDerivative = difReLU(Z1);
S = sum(sum(Hidden_output));

Del_nl=(Output-data).*difReLU(Z2);

Del_2 = (W2'*Del_nl).*hiddenDerivative;
% +beta*repmat(sparsity_vec,[1,size(data,2)])
A = W1*W1';
[U,E,V] = svd(A);
main_diag = 1./diag(E);
main_diag(isinf(main_diag))=0;
%K = V*diag(main_diag)*U';
temp = ones(hiddenSize,1)+main_diag;
divW1 = -1*V*diag((main_diag.*main_diag)/prod(temp))*U'*W1;


SA1 = softAbs(W1,alpha);
dSA1 = dSoftAbs(W1,alpha);

SA2 = softAbs(W2,alpha);
dSA2 = dSoftAbs(W2,alpha);

W1grad = (1/size(data,2))*Del_2 * data' + lambda1*dSA1 + beta*hiddenDerivative*data'+gamma*divW1;
b1grad = (1/size(data,2))*sum(Del_2,2) + beta*sum(hiddenDerivative,2);

W2grad = (1/size(Del_nl,2))*Del_nl * Hidden_output' + lambda2*dSA2;
b2grad = (1/size(Del_nl,2))*sum(Del_nl,2);

T1 = (norm(Output-data,'fro'))^2;
T2 = lambda1*(sum(sum(SA1)))+lambda2*(sum(sum(SA2)));
T3 = sum(log(temp));
cost = (1/(2*size(data,2)))* T1+ T2 + beta*S + (gamma/2)*T3;

disp([T1 T2 T3 S])

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 
