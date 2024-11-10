% clc 
% clear all
rho=0.1;
d=16;
theta_star = (A_h'*C_h^(-1)*A_h+rho*eye(d))^(-1)*A_h'*C_h^(-1)*B_h; %第一次调试时A_h忘记转置了
w_star = inv(C_h)*(-B_h+A_h*theta_star);
j_star = w_star'*(A_h*theta_star-B_h)-0.5*w_star'*C_h*w_star+0.5*rho*(theta_star'*theta_star);
tic
%%
nit = 4 * 1e4;%迭代步数
n = 6;% 智能体个数
rng(234)% 生成相同的随机数
sig = 0.5;%一致性步长
TT = 1:nit+1;
TT_j = 1:nit;
% theta梯度 gi(i为智能体数字) 20为theta维数d 20*10
g_theta_s = zeros(d*n,nit);
% w梯度 gi(i为智能体数字) 20为w维数d
g_w_s = zeros(d*n,nit);


%%
%时变拓扑
Wphi =  [ 0.5  0    0    0    0   0.5;
         0.5  0.5  0    0    0   0;
         0    0.5  0.5  1/3  0   0;
         0    0    0.5  1/3  0   0;
         0    0    0    1/3  0.5 0;
         0    0    0    0    0.5 0.5];%G1
%%
%次梯度_分布式
% rate_d = learnrate_doubling_trick(nit);%学习率,这个里面改成1/t
rate_d1 = 0.5;
rate_d2 = 0.02;
% Evaluatin of the convexcocave function (coincides with the evaluation of the Lagrangian under agreement of the multipliers)
lagrangian_k = zeros(nit, 1);

Dout   = diag( Wphi*ones(n,1) ); % This coincides with the identity
Lap    = Dout - Wphi; %L=D-A,计算L阵
P = eye(n)-sig*Lap;

Dout1  = diag( Wphi1*ones(n,1) ); % This coincides with the identity
Lap1    = Dout1 - Wphi1; %L=D-A,计算L阵
P1 = eye(n)-sig*Lap1;

%% 构造不同智能体不同的local reward矩阵

b1 = ones(n,1);
for i=1:2:n-2
    b1(i)=1-1/(3*i+1);
    b1(i+1)=2-b1(i);
end

% Primal and dual variables (s stands for saddle)
theta_s  = zeros(d*n, nit);
w_s = zeros(d*n,nit);
y_theta_s = zeros(d*n, nit);
y_w_s = zeros(d*n, nit);
z1 = ones(d*n, nit);
z2 = ones(d*n, nit);
%rng(0);
%tua= ones(1,nit);
%(Recall that each agent maintains a copy of the multiplier)
w_s_time_average = zeros(d*n,nit);
theta_s_time_average = zeros(d*n,nit);
j_average = ones(1,nit);
theta_ave = zeros(d,nit+1);
w_ave = zeros(d,nit+1);
e_average = ones(1,nit);

% auxiliary dual variable before computing ergodic sums

rate1(1:nit) = rate_d1;
rate2(1:nit) = rate_d2;
for t=1:5
    rate1(1:t)=1/sqrt(t);
    rate2(1:t) = 1/sqrt(t);
    % Subgradient (column vector) of Lagrangian with respect to primal
    % variable.
    %if ~mod(n,2)%非 取余 用来切换拓扑结构
    g_theta_s(:,t) = kron(eye(n),A_h')*w_s(:,t)+rho*theta_s(:,t);%生成梯度
    g_w_s(:,t) = kron(eye(n),A_h)*theta_s(:,t)-kron(eye(n),C_h)*w_s(:,t)-kron(b1,B_h);
 
    y_theta_s(:,t+1)= y_theta_s(:,t);
    y_w_s(:,t+1)= y_w_s(:,t);
    z1(:,t+1) =  z1(:,t); 
    z2(:,t+1) =  z2(:,t);
    
    theta_s(:,t+1) = -rate1(t)*y_theta_s(:,t+1)./z1(:,t+1);
    w_s(:,t+1) = -rate2(t)* y_w_s(:,t+1)./z2(:,t+1);
    
    theta_s_time_average(:,t+1) =  theta_s(:,t+1);
    w_s_time_average(:,t+1) =  w_s(:,t+1);
    
    for k=1:n
    j_average(k,t) = w_s_time_average(d*k-d+1:d*k,t)'*(A_h*theta_s_time_average(d*k-d+1:d*k,t)-B_h)-0.5*w_s_time_average(d*k-d+1:d*k,t)'*C_h*w_s_time_average(d*k-d+1:d*k,t)+0.5*rho*(theta_s_time_average(d*k-d+1:d*k,t)'*theta_s_time_average(d*k-d+1:d*k,t)); 
    end
    e_average(t)=abs((j_average(1,t)-j_star));%n个智能体MAPBE差值的均值
   
end

for t=6:nit
    rate1(1:t)=1/sqrt(t);
    rate2(1:t) = 1/sqrt(t);
    % Subgradient (column vector) of Lagrangian with respect to primal
    % variable.
    %if ~mod(n,2)%非 取余 用来切换拓扑结构
    g_theta_s(:,t) = kron(eye(n),A_h')*w_s(:,t)+rho*theta_s(:,t);%生成梯度
    g_w_s(:,t) = kron(eye(n),A_h)*theta_s(:,t)-kron(eye(n),C_h)*w_s(:,t)-kron(b1,B_h);
    
    tua(1,t)=randi([0,5]);
    y_theta_s(:,t+1)= kron(eye(d),Wphi)*y_theta_s(:,t-tua(1,t))+g_theta_s(:,t);
    y_w_s(:,t+1)= kron(eye(d),Wphi)*y_w_s(:,t-tua(1,t))-g_w_s(:,t);
    z1(:,t+1) =  kron(eye(d),Wphi)*z1(:,t-tua(1,t)); 
    z2(:,t+1) =  kron(eye(d),Wphi)*z2(:,t-tua(1,t));
    
    theta_s(:,t+1) = -rate1(t)*y_theta_s(:,t+1)./z1(:,t+1);
    w_s(:,t+1) = -rate2(t)* y_w_s(:,t+1)./z2(:,t+1);
    
    theta_s_time_average(:,t+1) =  theta_s(:,t+1);
    w_s_time_average(:,t+1) =  w_s(:,t+1);
    
    for k=1:n
    j_average(k,t) = w_s_time_average(d*k-d+1:d*k,t)'*(A_h*theta_s_time_average(d*k-d+1:d*k,t)-B_h)-0.5*w_s_time_average(d*k-d+1:d*k,t)'*C_h*w_s_time_average(d*k-d+1:d*k,t)+0.5*rho*(theta_s_time_average(d*k-d+1:d*k,t)'*theta_s_time_average(d*k-d+1:d*k,t)); 
    end
    e_average(t)=abs((j_average(1,t)-j_star));%n个智能体MAPBE差值的均值
   
end



for i = 1:n
    theta_ave(i,:) = theta_s(1+(i-1)*d,:);
end
theta_ave = sum(theta_ave)/n;
for i = 1:n
    w_ave(i,:) = w_s_time_average(1+(i-1)*d,:);
end
w_ave = sum(w_ave)/n;

theta=(theta_ave(1,:)-theta_star(1,:)).^2/n;

figure(1);
semilogy(TT,theta,'LineWidth',2);
title('(theta_{ave}-theta^*)^2/n');% figure(2);
% plot(TT,abs(w_ave(1,:)-w_star(1,:)),'LineWidth',2);
% title('|w-wstar|/|wtar|');
toc
figure(3);
semilogy(TT_j,e_average,'LineWidth',2);
ylabel('$\frac{|J-J^*|}{J^*}$','Interpreter','latex');
xlabel('$time/s$','Interpreter','latex');
%set(gca,'XLim',[0 1000]);
%legend('$\eta=\frac{2}{5000^{0.55}}$','$\eta=\frac{2.5}{5000^{0.55}}$','$\eta=\frac{3}{5000^{0.55}}$','Interpreter','latex')