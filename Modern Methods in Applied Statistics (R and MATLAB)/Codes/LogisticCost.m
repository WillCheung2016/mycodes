function [cost,grad] = LogisticCost(theta, inputSize, numClasses, data, labels_vec, lambda)
%Mask = ones(hiddenSize2, hiddenSize1+1);

W = reshape(theta(1:numClasses*inputSize), numClasses, inputSize);
b = reshape(theta(numClasses*inputSize+1:end), numClasses, 1);

N = size(data,2);

groundTruth = full(sparse(labels_vec, 1:N, 1));

Powers = W*data+repmat(b,[1,N]);
alpha = max(Powers,[],1);
Powers = bsxfun(@minus,Powers,alpha) ;
Exp_Mat = exp(Powers);
Sum_Vec = sum(Exp_Mat);
log_sum = 0;
for i = 1:N
    j = labels_vec(i);
    log_sum = log_sum + log(Exp_Mat(j,i)/Sum_Vec(i));
end
cost = (-1/N)*log_sum + (lambda/2)*(norm(W,'fro')^2);

Del_nl= Exp_Mat./repmat(Sum_Vec,[numClasses,1]) - groundTruth;
Wgrad = (1/N) * Del_nl * data' + lambda*W;
bgrad = (1/N) * sum(Del_nl,2);

grad = [Wgrad(:) ; bgrad(:)];

end
