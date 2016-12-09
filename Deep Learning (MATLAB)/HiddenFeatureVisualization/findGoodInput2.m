% This function performs gradient ascend algorithm to find the input vector
% which can maximize the activation of a neuron in the second layer given
% its corresponding weight vector, bias, and the weight matrix and bias
% vector of the first layer.

function [vector, value]= findGoodInput2(w,b,W1,b1,iter)

L = size(W1,2);
x = rand(L,1);
z1 = W1*x+b1;
z2 = w'*sigmoid(z1)+b;
gradx = sigmoid(z2)*(1-sigmoid(z2))*W1'*diag(sigmoid(z1).*(1-sigmoid(z1)))*w;
x = x/norm(x);
for i = 1:iter
    x = x + gradx;
    if norm(x,2)>1
        x = x/norm(x,2);
    end
    z1 = W1*x+b1;
    z2 = w'*sigmoid(z1)+b;
    %vals(i) = sigmoid(z2);
    gradx = sigmoid(z2)*(1-sigmoid(z2))*W1'*diag(sigmoid(z1).*(1-sigmoid(z1)))*w;
end

vector = x;
value = sigmoid(z2);
end

function [cost,grad] = hiddenNeuronCost(x,w,b,W1,b1)
z1 = W1*x+b1;
z2 = w'*sigmoid(z1)+b;
grad = -1*sigmoid(z2)*(1-sigmoid(z2))*W1'*diag(sigmoid(z1).*(1-sigmoid(z1)))*w;
cost = -1*sigmoid(z2);
end
