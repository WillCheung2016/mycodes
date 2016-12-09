function [WhitenA, ZCAWhite] = ZCAwhiten(A,epsilon)
% This function computes the whitening matrix and whitened
% data given raw data and smoothing parameter epsilon.

meanPatch = mean(A, 2);  
patches = bsxfun(@minus, A, meanPatch);
numPatches = size(A,2);
Sigma = patches * patches' / numPatches;
[U, S, V] = svd(Sigma);
ZCAWhite = U * diag(1 ./ sqrt(diag(S) + epsilon)) * U';
WhitenA = ZCAWhite * A;

end
