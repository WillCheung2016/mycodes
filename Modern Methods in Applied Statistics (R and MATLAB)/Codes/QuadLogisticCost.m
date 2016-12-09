function [cost,grad] = QuadLogisticCost(theta, inputSize, numClasses, data, labels_vec, lambda)
%Mask = ones(hiddenSize2, hiddenSize1+1);

[W, b, classCov] = extractLogisticWeights(theta, inputSize, numClasses);
N = size(data,2);

difClassCov = zeros(inputSize,inputSize,numClasses);

linTerm = W*data+repmat(b,[1,N]);
quadTerm = zeros(N,numClasses);

sum_cov_norms = 0;

for i = 1:numClasses
    quadTerm(:,i) = diag(data'*classCov(:,:,i)*data);
    sum_cov_norms = sum_cov_norms + norm(classCov(:,:,i),'fro')^2;
end
% Z3 = W3 * Hidden_output + repmat(b3,[1,N]);
% Output=sigmoid(Z3);

quadTerm = quadTerm';

groundTruth = full(sparse(labels_vec, 1:N, 1));

Powers = linTerm + quadTerm;
alpha = max(Powers,[],1);
Powers = bsxfun(@minus,Powers,alpha) ;
Exp_Mat = exp(Powers);
Sum_Vec = sum(Exp_Mat);
log_sum = 0;
for i = 1:N
    j = labels_vec(i);
    log_sum = log_sum + log(Exp_Mat(j,i)/Sum_Vec(i));
end
cost = (-1/N)*log_sum + (lambda/2)*(sum_cov_norms + norm(W,'fro')^2);

Del_nl= Exp_Mat./repmat(Sum_Vec,[numClasses,1]) - groundTruth;
Wgrad = (1/N) * Del_nl * data' + lambda*W;
bgrad = (1/N) * sum(Del_nl,2);

for i = 1:numClasses
    difClassCov(:,:,i) = (1/N) * data*diag(Del_nl(i,:))*data' + lambda*classCov(:,:,i);
end

grad = [Wgrad(:) ; bgrad(:) ; difClassCov(:)];

end
