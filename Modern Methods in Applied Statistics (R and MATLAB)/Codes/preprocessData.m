function [group_cov,numInGroups] = preprocessData(data, groupings,inputSize, numClasses)

group_cov = zeros(inputSize,inputSize,numClasses);
numInGroups = zeros(numClasses,1);

for i = 1:numClasses
    ind = find(groupings==i);
    N = length(ind);
    numInGroups(i) = N;
    group_data = data(:,ind);
    group_cov(:,:,i) = ((N-1)/N)*cov(group_data');
    
end

end