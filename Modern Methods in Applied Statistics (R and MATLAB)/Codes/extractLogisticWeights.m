function [W, b, classCov] = extractLogisticWeights(theta, inputSize, numClasses)
W = reshape(theta(1:numClasses*inputSize), numClasses, inputSize);
b = reshape(theta(numClasses*inputSize+1:numClasses*inputSize+numClasses), numClasses, 1);
temp = theta(numClasses*inputSize+numClasses+1:end);
classCov = reshape(temp,[inputSize,inputSize,numClasses]);

end