function [softmaxModel] = softmaxTrain(inputSize, numClasses, max_norm, inputData, labels, options)
%softmaxTrain Train a softmax model with the given parameters on the given
% data. Returns softmaxOptTheta, a vector containing the trained parameters
% for the model.
%
% inputSize: the size of an input vector x^(i)
% numClasses: the number of classes 
% lambda: weight decay parameter
% inputData: an N by M matrix containing the input data, such that
%            inputData(:, c) is the cth input
% labels: M by 1 matrix containing the class labels for the
%            corresponding inputs. labels(c) is the class label for
%            the cth input
% options (optional): options
%   options.maxIter: number of iterations to train for

if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 2;
end

% initialize parameters
thetaW = 0.005 * randn(numClasses * inputSize, 1);
thetab = zeros(numClasses, 1);
theta = [thetaW(:) ; thetab(:)];
% Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % softmaxCost.m satisfies this.
minFuncOptions.display = 'on';

for i = 1:500
    [theta, cost] = minFunc( @(p) softmaxCost(p, ...
                                   numClasses, inputSize, ...
                                   inputData, labels), ...                                   
                              theta, options);
    W1 = reshape(theta(1:inputSize*numClasses), numClasses, inputSize);
    norm_W1 = sqrt(sum(W1.*W1,2));
    rescale_ind = find(norm_W1 > max_norm);
    
    W1(rescale_ind,:) = max_norm * W1(rescale_ind,:)./repmat(norm_W1(rescale_ind),[1, size(W1,2)]);
    theta(1:inputSize*numClasses) = W1(:);
end
% Fold softmaxOptTheta into a nicer format
softmaxModel.optTheta = theta;
softmaxModel.inputSize = inputSize;
softmaxModel.numClasses = numClasses;
                          
end                          
