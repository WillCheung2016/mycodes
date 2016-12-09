% This function performs gradient ascend algorithm to find the input vector
% which can maximize the activation of a neuron in the first layer given
% its corresponding weight vector. Actually this weight vector is the input
% vector. This function is written for fun.

function [vector, value]= findGoodInput1(w,b,inputSize,iter)

L = inputSize;
x = rand(L,1);
z1 = w'*x+b;
gradx = sigmoid(z1)*(1-sigmoid(z1))*w;
x = x/norm(x);
for i = 1:iter
    x = x + gradx;
    x = x/norm(x,2);
    z1 = w'*x+b;
    %vals(i) = sigmoid(z2);
    gradx = sigmoid(z1)*(1-sigmoid(z1))*w;
end

vector = x;
value = sigmoid(z1);
end


