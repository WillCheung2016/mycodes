% This function tries to find the input vector which can maximize the
% activation of a hidden neuron in the second hidden layer by lease square
% with positivity constraint, given the weight and bias vector of the
% neuron, and the weight matrix and bias vector of the first layer.

function [vector, value]= findGoodInput2_ContLeastSq(w,b,W1,b1,eps)

options.Display = 'off';
hiddenSize1 = size(W1,1);
inputSize1 = size(W1,2);
f=lsqlin(w',log(99)-b,[],[],[],[],eps*ones(hiddenSize1,1),(1-eps)*ones(hiddenSize1,1),[],options);

z1=log(f./(1-f));

x=lsqlin(W1,z1-b1,[],[],[],[],0.5*eps*ones(inputSize1,1),(1-eps)*ones(inputSize1,1),[],options);

%x = x/norm(x);

vector = x;
value = sigmoid(w'*sigmoid(W1*x+b1)+b);
end
