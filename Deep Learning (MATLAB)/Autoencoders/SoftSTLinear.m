function y = SoftSTLinear(x, base)
y = (newSoftplus(x+1,base) - newSoftplus(x-1,base))/2;
end