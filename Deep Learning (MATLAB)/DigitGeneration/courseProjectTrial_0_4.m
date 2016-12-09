% This script reads decoding weight matrices of each layer from a   
% 3-layer autoencoder trained on MNIST, and generates digits based on
% the user's input.

% Read Data from .mat files.
clear
load('ZCAall_opttheta.mat')
[D1,b1] = extractDecodingMatrix(opttheta, 200, 28*28);
load('ZCAall_opttheta2.mat')
[D2,b2] = extractDecodingMatrix(opttheta2, 50, 200);
load('ZCAall_opttheta3.mat')
[D3,b3] = extractDecodingMatrix(opttheta3, 10, 50);

% index   number
%   1       9
%   2       8
%   3       6 
%   4       9
%   5       5
%   6       6
%   7       7
%   8       7
%   9       5
%   10      8

% Standard deviations of Noise added to different layers as digit
% generating process.
sig_1 = 0.1;
sig_2 = 0.05;
sig_3 = 0.05;
zero = [5,6,7];
one = [3,8,9];
two = 1;
three = 2;
four = [4,10];
quit = 0;
while(1)
    prompt = 'Enter a number from 0-4? Quit if any input other than 5 digits.';
    x = input(prompt);
    
    count=0;       
    while(count<100)
        level1_perturb = (sig_1)^2*randn(200,1);
        level2_perturb = (sig_2)*2*randn(50,1);
        level3_perturb = (sig_3)*2*randn(10,1);
        num_indicator=zeros(10,1);
        if (x==0)
            ind = randi(length(zero),1);
            num_indicator(zero(ind))=2;
        elseif (x==1)
            ind = randi(length(one),1);
            num_indicator(one(ind))=2;
        elseif (x==2)
            ind = randi(length(two),1);
            num_indicator(two(ind))=2;
        elseif (x==3)
            ind = randi(length(three),1);
            num_indicator(three(ind))=2;
        elseif (x==4)
            ind = randi(length(four),1);
            num_indicator(four(ind))=2;
        else
            quit = 1;
        end
        if (quit == 1)
            break
        end
        generated_number = sigmoid(D1*(sigmoid(D2*(sigmoid(D3*(num_indicator+level3_perturb)+b3)+level2_perturb)+b2)+level1_perturb)+b1);
        display_network(generated_number);
        count = count+1;
    end
    if (quit == 1)
        break
    end
end

