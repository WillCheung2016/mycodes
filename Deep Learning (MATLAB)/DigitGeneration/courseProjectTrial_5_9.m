% This script reads decoding weight matrices of each layer from a   
% 3-layer autoencoder trained on MNIST, and generates digits based on
% the user's input.

% Read Data from .mat files.
clear
load('ZCA5-9_opttheta.mat')
[D1,b1] = extractDecodingMatrix(opttheta, 200, 28*28);
load('ZCA5-9_opttheta2.mat')
[D2,b2] = extractDecodingMatrix(opttheta2, 50, 200);
load('ZCA5-9_opttheta3.mat')
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
sig_1 = 0.1;
sig_2 = 0.05;
sig_3 = 0.05;
five = [5,9];
six = [3,6];
seven = [7,8];
eight = [2,10];
nine = [1,4];

num_indicator=zeros(10,1);
num_indicator(10)=2;
quit = 0;
while(1)
    prompt = 'Enter a number from 5-9? Quit if any input other than 5 digits.';
    x = input(prompt);
    
    count=0;       
    while(count<100)
        level1_perturb = (sig_1)^2*randn(200,1);
        level2_perturb = (sig_2)*2*randn(50,1);
        level3_perturb = (sig_3)*2*randn(10,1);
        num_indicator=zeros(10,1);
        if (x==5)
            ind = randi(length(five),1);
            num_indicator(five(ind))=2;
        elseif (x==6)
            ind = randi(length(six),1);
            num_indicator(six(ind))=2;
        elseif (x==7)
            ind = randi(length(seven),1);
            num_indicator(seven(ind))=2;
        elseif (x==8)
            ind = randi(length(eight),1);
            num_indicator(eight(ind))=2;
        elseif (x==9)
            ind = randi(length(nine),1);
            num_indicator(nine(ind))=2;
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

