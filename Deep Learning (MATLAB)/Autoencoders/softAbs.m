function output = softAbs(x,alpha)
    temp = alpha*x;
    output = (1/alpha)*(Softplus(temp)+Softplus(-1*temp));
end