function sigm = sigmoid(x)
    %x = gather(x);
    sigm = 1 ./ (1 + exp(-x));
    %sigm = gpuArray(sigm);
end