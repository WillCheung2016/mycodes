function [pred] = softmaxPredict(softmaxModel, data)

numClasses = softmaxModel.numClasses;
inputSize = softmaxModel.inputSize;
theta = softmaxModel.optTheta;
W1 = reshape(theta(1:inputSize*numClasses), numClasses, inputSize);
b1 = theta(inputSize*numClasses+1:end);

pred = zeros(1, size(data, 2));

[Max_Prob, pred] = max(W1*data + repmat(b1,[1,size(data,2)]),[],1); 








% ---------------------------------------------------------------------

end

