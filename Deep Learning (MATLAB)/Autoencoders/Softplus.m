function output = Softplus(x)
    %x = gather(x);
    M = x>700;
    output = log(1+exp((~M).*x)) - M.*log(2) + M.*x;
    %output = gpuArray(output);
end