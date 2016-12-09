function theta = quadLogistic_initialization(inputSize, numClasses)

r  = sqrt(6) / sqrt(numClasses+inputSize+1);
W = rand(numClasses, inputSize) * 2 * r - r;
b = zeros(numClasses, 1);

classCov = zeros(inputSize,inputSize,numClasses);

for i =1:numClasses
    classCov(:,:,i) = eye(inputSize);
end

theta = [W(:) ; b(:) ; classCov(:)];

end