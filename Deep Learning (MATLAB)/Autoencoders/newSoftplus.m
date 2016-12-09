function output = newSoftplus(x,base)
    %x = gather(x);
    M = x>(700/log(base));
    output = log(1+base.^((~M).*x))/log(base) - M.*(log(2)/log(base)) + M.*x;
    %output = gpuArray(output);
end