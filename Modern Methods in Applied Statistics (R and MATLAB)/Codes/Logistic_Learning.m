clear

temp=load('proc_phoneme.csv');
load('ind_matrix.csv')
data = temp(:,1:256)';
groupings = temp(:,257)';
C = zeros(5,5,5);

inputSize = size(data,1);
numClasses = max(groupings);
lambda = 3e-2;
N = size(data,2);
% best lambda 3e-2


addpath minFunc/
options.Method = 'lbfgs';
options.MaxFunEvals = 1000;                          
options.maxIter = 200;	  
options.display = 'on';

AllMSEs = zeros(5,1);
AllMSPEs = zeros(5,1);
all_ind = 1:N;

for k = 1 : 5
    
    train_data = data(:,ind_matrix(:,k));
    train_groupings = groupings(ind_matrix(:,k));

    te_ind = setdiff(all_ind,ind_matrix(:,k));
    test_data = data(:,te_ind);
    test_groupings = groupings(te_ind);
    
    mu = mean(train_data,2);
    sigma2 = var(train_data,0,2);
    sigma = sqrt(sigma2);
    train_data = (train_data - repmat(mu,[1,size(train_data,2)]))./repmat(sigma,[1,size(train_data,2)]);
    test_data = (test_data - repmat(mu,[1,size(test_data,2)]))./repmat(sigma,[1,size(test_data,2)]);
    
    theta = Logistic_initialization(inputSize, numClasses);
    
    [theta, cost] = minFunc( @(p) LogisticCost(p, ...
        inputSize, numClasses, train_data, train_groupings,lambda), ...
        theta, options);
    
    pred = LogisticPredict(theta, inputSize, numClasses, train_data);
    acc = mean(train_groupings(:) == pred(:));
    disp(acc)
    AllMSEs(k) = 1-acc;
    
    pred = LogisticPredict(theta, inputSize, numClasses, test_data);
    acc = mean(test_groupings(:) == pred(:));
    disp(acc)
    C(:,:,k) = confusionmat(test_groupings,pred);
    AllMSPEs(k) = 1-acc;
end
disp(AllMSPEs)