%clear

temp=load('vehicle3.txt');

data = temp(:,1:18)';
groupings = temp(:,19)';

inputSize = size(data,1);
numClasses = max(groupings);

[group_cov,numInGroups] = preprocessData(data, groupings,inputSize, numClasses);

theta = parameters_initialization(inputSize, numClasses, group_cov, data);
[B0, Adas0] = extractParameters(theta, inputSize, numClasses);

addpath minFunc/
options.Method = 'scg';
options.MaxFunEvals = 1000;                          
options.maxIter = 2;	  
options.display = 'off';

delta = 1;
while delta>1e-5
    [theta, cost] = minFunc( @(p) CPCAcost(p, ...
            group_cov, numInGroups, inputSize, numClasses), ...
            theta, options);
    B = reshape(theta(1:inputSize^2),[inputSize,inputSize]);
    [U,S,V] = svd(B'*B);
    diag_vec = sqrt(diag(S));    
    new_B = B*(U*diag(1./diag_vec)*U');
    theta(1:inputSize^2) = new_B(:);
    delta = max(max(abs(B-new_B)));
    disp(delta) 
end
    
    
[B, Adas] = extractParameters(theta, inputSize, numClasses);


