function [pred] = QuadLogisticPredict(theta, inputSize, numClasses, data)

[W, b, classCov] = extractLogisticWeights(theta, inputSize, numClasses);
N = size(data,2);

linTerm = W*data+repmat(b,[1,N]);
quadTerm = zeros(N,numClasses);

for i = 1:numClasses
    quadTerm(:,i) = diag(data'*classCov(:,:,i)*data);
end

Powers = linTerm + quadTerm';

[Max_Prob, pred] = max(Powers, [], 1);

end
