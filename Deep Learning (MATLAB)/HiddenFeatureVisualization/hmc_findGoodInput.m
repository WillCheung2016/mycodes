% This function performs hamitonian monte carlo algorithm to find the input
% vector which can maximize the activation of a neuron in the second layer
% given its corresponding weight vector, bias, and the weight matrix and bias
% vector of the first layer. This function is written for fun.

function [vector, value]= hmc_findGoodInput(w,b,W1,b1,iter,LF,rho)

L = size(W1,2);
x_current = rand(L,1);
%x_current = mu;
%x = 2*rand(L,1)-1;
%x_current = mean(W1)';

g = computeDelta(x_current,w,b,W1,b1);
E = computeOutput(x_current,w,b,W1,b1);
for i = 1:iter
    p = randn(L,1);
    H = p'*p/2 + E;

    xnew = x_current;
    gnew = g;
    for j = 1:LF
        p = p - rho * gnew / 2;
        xnew = xnew + rho * p;
        gnew = computeDelta(xnew,w,b,W1,b1);
        p = p - rho * gnew /2;
    end
    if (max(abs(xnew))>1)
        xnew = xnew/max(abs(xnew));
    end
    Enew = computeOutput(xnew,w,b,W1,b1);
    Hnew = p'*p/2 + Enew;
    dH = Hnew - H;
    % && (norm(xnew,2)<=1)
    if ((dH<0))
        accept = 1;
    elseif ((rand(1)<exp(-1*dH)))
        accept = 1;
    else
        accept = 0;
    end
    
    if (accept)
        g = gnew;
        x_current = xnew;
        E = Enew;
        %disp('Accepted !')
    end
    display_network(x_current);
end

vector = x_current;
value = computeOutput(x_current,w,b,W1,b1);
end

function grad = computeDelta(x,w,b,W1,b1)
z1 = W1*x+b1;
z2 = w'*sigmoid(z1)+b;
grad = (-1)*(1/sigmoid(z2))*sigmoid(z2)*(1-sigmoid(z2))*W1'*diag(sigmoid(z1).*(1-sigmoid(z1)))*w;
end

function px = computeOutput(x,w,b,W1,b1)
z1 = W1*x+b1;
z2 = w'*sigmoid(z1)+b;
px = (-1)*log(sigmoid(z2));
end
