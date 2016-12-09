function theta = Logistic_initialization(inputSize, numClasses)

r  = sqrt(6) / sqrt(numClasses+inputSize+1);
W = rand(numClasses, inputSize) * 2 * r - r;
b = zeros(numClasses, 1);

theta = [W(:) ; b(:)];

end