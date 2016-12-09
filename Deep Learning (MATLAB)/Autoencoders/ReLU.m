function output = ReLU(x)
    M = x>0;
    output = M.*x;
end