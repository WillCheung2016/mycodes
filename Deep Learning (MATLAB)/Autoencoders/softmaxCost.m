function [cost, grad] = softmaxCost(theta, numClasses, inputSize, data, labels)


W1 = reshape(theta(1:inputSize*numClasses), numClasses, inputSize);
b1 = theta(inputSize*numClasses+1:end);
numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));

Powers = W1*data + repmat(b1,[1,numCases]);
alpha = max(Powers,[],1);
Powers = bsxfun(@minus,Powers,alpha) ;
Exp_Mat = exp(Powers);
Sum_Vec = sum(Exp_Mat);
log_sum = 0;
for i = 1:numCases
    j = labels(i);
    log_sum = log_sum + log(Exp_Mat(j,i)/Sum_Vec(i));
end
cost = (-1/numCases)*log_sum;

Del_nl = Exp_Mat./repmat(Sum_Vec, [numClasses, 1]) - groundTruth;
W1grad = (1/numCases)*Del_nl * data';
b1grad = (1/numCases)*sum(Del_nl, 2);

grad = [W1grad(:) ; b1grad(:)];

end

