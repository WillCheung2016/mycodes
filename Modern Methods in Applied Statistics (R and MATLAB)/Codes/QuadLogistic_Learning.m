clear

temp=load('proc_phoneme.csv');
load('ind_matrix.csv')
data = temp(:,1:256)';
groupings = temp(:,257)';

inputSize = size(data,1);
numClasses = max(groupings);
lambda = 1e-4;
N = size(data,2);
epoch = 30;
batchSize = 200;
% Additive lambda=3e-4 sigma=0.15

addpath minFunc/
options.Method = 'lbfgs';
options.MaxFunEvals = 1000;                          
options.maxIter = 2;	  


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
%     
%     [train_data, ZCAWhiten] = ZCAwhiten(train_data,1e-4);
%     test_data = ZCAWhiten*test_data;
    
    theta = quadLogistic_initialization(inputSize, numClasses);
    options.display = 'off';
    for i = 1:epoch
        j = 1;
        if (i>epoch-2)
            options.display = 'on';
        end
        while j~=N
            cor_data = train_data(:,j:min([j+batchSize-1,N]));
            %cor_mat = 1 + 0.3*randn(size(cor_data));
            % binornd(1,0.8, size(cor_data,1),size(cor_data,2))
            cor_data = cor_data + 0.15*randn(size(cor_data));
            temp_groupings = train_groupings(j:min([j+batchSize-1,N]));
            [theta, cost] = minFunc( @(p) QuadLogisticCost(p, ...
                inputSize, numClasses, cor_data, temp_groupings,lambda), ...
                theta, options);
            j = min([j + batchSize, N]);
        end
    end
    pred = QuadLogisticPredict(theta, inputSize, numClasses, train_data);
    acc = mean(train_groupings(:) == pred(:));
    disp(acc)
    AllMSEs(k) = 1-acc;
    
    pred = QuadLogisticPredict(theta, inputSize, numClasses, test_data);
    acc = mean(test_groupings(:) == pred(:));
    disp(acc)
    AllMSPEs(k) = 1-acc;
end
