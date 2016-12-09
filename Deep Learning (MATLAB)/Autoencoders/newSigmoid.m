function sigm = newSigmoid(x,base)
    %x = gather(x);
    sigm = 1 ./ (1 + exp(-x*log(base)));
    %sigm = gpuArray(sigm);
end