function [B, Adas] = extractParameters(theta, inputSize, numClasses)

B = reshape(theta(1:inputSize^2),[inputSize,inputSize]);
Adas = reshape(theta(inputSize^2+1:inputSize^2+inputSize*numClasses),[inputSize,numClasses]);
% M_diag = theta(inputSize^2+inputSize*numClasses+1:inputSize^2+inputSize*numClasses+inputSize);
% M_offdiag = theta(inputSize^2+inputSize*numClasses+inputSize+1:end);
% 
% temp = zeros(inputSize,inputSize);
% temp(eye(inputSize)==1) = M_diag;
% 
% M = temp + squareform(M_offdiag);

end