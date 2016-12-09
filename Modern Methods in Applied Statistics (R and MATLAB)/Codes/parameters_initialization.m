function theta = parameters_initialization(inputSize, numClasses,group_cov, data)

N = size(data,2);
pop_cov = ((N-1)/N)*cov(data');
B = orth(pop_cov);
%B = orth(randn(inputSize,inputSize));
%Adas = ones(inputSize,numClasses);
Adas = zeros(inputSize,numClasses);

for i = 1:numClasses
    [U,S,V] = svd(group_cov(:,:,i));
    Adas(:,i) = diag(S);
end
% M_diag = rand(inputSize,1);
% M_offdiag = rand(nchoosek(inputSize,2),1);

theta = [B(:) ; Adas(:)];