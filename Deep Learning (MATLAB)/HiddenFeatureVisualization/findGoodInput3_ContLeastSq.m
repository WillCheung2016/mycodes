% This function tries to find the input vector which can maximize the
% activation of a hidden neuron in the third hidden layer by lease square
% with positivity constraint, given the weight and bias vector of the
% neuron, and the weight matrices and bias vectors of the first and the 
% second layers.

function [vector, value]= findGoodInput3_ContLeastSq(w,b,W1,b1,W2,b2,eps)

options.Display = 'off';
hiddenSize1 = size(W1,1);
inputSize1 = size(W1,2);

hiddenSize2 = size(W2,1);

a2=lsqlin(w',log(9)-b,[],[],[],[],eps*ones(hiddenSize2,1),(1-eps)*ones(hiddenSize2,1),[],options);

z2=log(a2./(1-a2));

a1=lsqlin(W2,z2-b2,[],[],[],[],0.5*eps*ones(hiddenSize1,1),(1-eps)*ones(hiddenSize1,1),[],options);

z1=log(a1./(1-a1));

x=lsqlin(W1,z1-b1,[],[],[],[],0.5*eps*ones(inputSize1,1),(1-eps)*ones(inputSize1,1),[],options);

x = x/norm(x);

vector = x;
value = sigmoid(w'*sigmoid(W2*sigmoid(W1*x+b1)+b2)+b);
end
