function [cost, grad] = CPCAcost(theta,  group_cov, numInGroups, inputSize, numClasses)

% All groupings are encoded by positive integers

B = reshape(theta(1:inputSize^2),[inputSize,inputSize]);
Adas = reshape(theta(inputSize^2+1:inputSize^2+inputSize*numClasses),[inputSize,numClasses]);
% M_diag = theta(inputSize^2+inputSize*numClasses+1:inputSize^2+inputSize*numClasses+inputSize);
% M_offdiag = theta(inputSize^2+inputSize*numClasses+inputSize+1:end);

% temp = zeros(inputSize,inputSize);
% temp(eye(inputSize)==1) = M_diag;
% M = temp + squareform(M_offdiag);

numInGroups = numInGroups-1;

accum = 0;
dB = zeros(size(B));
dAdas = zeros(inputSize,numClasses);
for i = 1:numClasses
    inv_diag_vec = 1./Adas(:,i);
    inv_diag_mat = diag(1./Adas(:,i));
    accum = accum + numInGroups(i)*(sum(log(Adas(:,i))) + trace(group_cov(:,:,i)*B*inv_diag_mat*B'));
    dB = dB + numInGroups(i)*(2*group_cov(:,:,i)*B*inv_diag_mat);
    dAdas(:,i) = numInGroups(i)*(inv_diag_vec - diag(B'*group_cov(:,:,i)*B).*(inv_diag_vec.^2));
end

cost = accum;

% + sum(sum(M.*(B'*B-eye(inputSize))))
% dB = dB + 2*B*M;

% dM = B'*B - eye(inputSize);
% dM_diag = diag(dM);
% dM(eye(inputSize)==1) = zeros(inputSize,1);
% dM_offdiag = 2*squareform(dM);

grad = [dB(:) ; dAdas(:)];

    



