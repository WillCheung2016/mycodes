% This function performs gradient ascend algorithm to find the input vector
% which can maximize the activation of a neuron in the third hidden layer 
% given its corresponding weight vector, bias, and the weight matrices and
% bias vectors of the first and the second layer.

function [vector, value]= findGoodInput3(w,b,W1,b1,W2,b2,iter)

L = size(W1,2);
x = rand(L,1);
%x = mean(W1)';
%%---------------------------------
%%options.Method = 'lbfgs';                        
%%options.maxIter = 400;	  
%%options.display = 'on';
%%options.TolFun = 1e-10;
%%options.TolX = 1e-10;
%%for i = 1:iter
%%    [x, cost] = minFunc( @(p) hiddenNeuronCost(p, w, b,W1, b1), x, options);
%%    x = x/norm(x);
%%end
%%---------------------------------
z1 = W1*x+b1;
z2 = W2*sigmoid(z1)+b2;
z3 = w'*sigmoid(z2)+b;
gradx = W1'*diag(sigmoid(z1).*(1-sigmoid(z1)))*W2'*diag(sigmoid(z2).*(1-sigmoid(z2)))*w*sigmoid(z3)*(1-sigmoid(z3));
x = x/norm(x);
%vals = zeros(1,iter);
for i = 1:iter
    x = x + gradx;
    if norm(x)>1
        x = x/norm(x,2);
    end
    z1 = W1*x+b1;
    z2 = W2*sigmoid(z1)+b2;
    z3 = w'*sigmoid(z2)+b;
    %vals(i) = sigmoid(z3);
    gradx = W1'*diag(sigmoid(z1).*(1-sigmoid(z1)))*W2'*diag(sigmoid(z2).*(1-sigmoid(z2)))*w*sigmoid(z3)*(1-sigmoid(z3));
end

vector = x;
value = sigmoid(z3);
end

function [cost,grad] = hiddenNeuronCost(x,w,b,W1,b1)
z1 = W1*x+b1;
z2 = w'*sigmoid(z1)+b;
grad = -1*sigmoid(z2)*(1-sigmoid(z2))*W1'*diag(sigmoid(z1).*(1-sigmoid(z1)))*w;
cost = -1*sigmoid(z2);
end
