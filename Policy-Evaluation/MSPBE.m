% 初始声明
clear all
load('C:\Users\zyy\Desktop\Distributed_Policy_Evaluation_over_Multiagent_Network_with_Communication_Delays\results\trial_5_nrbf_20_alpha_0.100000.mat');
num_rbfs = 4;                   % number of RBFs for one state, total number of RBFs: N^state_dim * action_dim 一个状态的径向基函数的个数，总的径向基函数的个数为N^state_dim * action_dim
[centroids, dev] = BuildStateList(num_rbfs);  % the center of the RBFs and their deviations are created建立径向基函数的中心和偏差
new_state = state_action_r(1:422,1:2);
new_state_action=state_action_r(1:422,1:3);
new_action = state_action_r(1:422,3);
new_reward = state_action_r(1:422,4);
% new_reward = rand(422,1);
% new_reward(422,1) = 5.5;
new_phi = zeros(num_rbfs^2, 1);
newnew_phi = zeros(num_rbfs^2, 1);
% A_h = zeros(length(centroids), length(centroids));
% B_h = zeros(length(centroids), 1);
% C_h = zeros(length(centroids), length(centroids));
rng(234)% 生成相同的随机数
% H =15*rand(20,2);
% rng(234)% 生成相同的随机数
% H1 =2*rand(25,2);
% H2 =rand(25,2);
% H3 =rand(25,2);
A_h = zeros(num_rbfs^2,num_rbfs^2);
B_h = zeros(num_rbfs^2, 1);
C_h = zeros(num_rbfs^2, num_rbfs^2);
% 
% for i = 1:421
%     if new_reward(i)<0.5
%         new_reward(i)=0;
%     else
%         new_reward(i)=new_reward(i);
%     end
% end
        

 for i = 1:421
     new_phi = phi(new_state(i,:),centroids,dev);%state为行向量
     newnew_phi = phi(new_state(i+1,:),centroids,dev);%下一时刻的phi值
     
%      new_phi=phi(new_state_action(i,:),new_action(i),centroids,dev);
%      newnew_phi = phi(new_state_action(i+1,:),new_action(i+1),centroids,dev);
     
%      new_phi = 0.5*cos(H*new_state(i,:)');
%      newnew_phi = 0.5*cos(H*new_state(i+1,:)');
%      new_phi = phi(new_state(i,:), centroids);
%      newnew_phi = phi(new_state(i+1,:), centroids);
         
     A = new_phi*(new_phi-0.9*newnew_phi)';
     A_h = A_h + A;
     B = new_reward(i)*new_phi;
     B_h = B_h + B;
     C = new_phi*new_phi';
     C_h = C_h + C;
 end
 A_h = A_h/421;
 B_h = B_h/421;
 C_h = C_h/421;
 
% file_name = 'results/test12.mat';
% save(file_name, 'A_h', 'B_h', 'C_h');
     
     
     
