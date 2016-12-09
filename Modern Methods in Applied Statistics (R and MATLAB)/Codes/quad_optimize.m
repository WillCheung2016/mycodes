Q=-2*classCov(:,:,5);
f = -1*W(5,:)';
c = -1*b(5);

H{1} = 2*eye(inputSize);
kk{1} = zeros(inputSize,1);
dd{1} = 0;

fun = @(x)quadobj(x,Q,f,c);
nonlconstr = @(x)quadconstr(x,H,kk,dd);

options = optimoptions(@fmincon,'Algorithm','interior-point',...
    'GradObj','on','GradConstr','on','Hessian','user-supplied',...
    'HessFcn',@(x,lambda)quadhess(x,lambda,Q,H));

x0 = zeros(inputSize,1); % column vector
[x,fval,eflag,output,lambda] = fmincon(fun,x0,...
    [],[],[],[],[],[],nonlconstr,options);