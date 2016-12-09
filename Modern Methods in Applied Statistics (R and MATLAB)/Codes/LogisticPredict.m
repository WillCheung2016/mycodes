function [pred] = LogisticPredict(theta, inputSize, numClasses, data)

W = reshape(theta(1:numClasses*inputSize), numClasses, inputSize);
b = reshape(theta(numClasses*inputSize+1:end), numClasses, 1);
N = size(data,2);

Powers = W*data+repmat(b,[1,N]);

[Max_Prob, pred] = max(Powers, [], 1);

end
