function [ centers, dev ] = BuildStateList(num_rbfs)
% Generates the centers and dev of the basis functions
x = linspace(0, 1, num_rbfs); % normalized state space 产生0-1间num_rbfs点行矢量 状态空间
xp = linspace(0, 1, num_rbfs);

centers = zeros(length(x) * length(xp), 2);%前一时刻状态和后一时刻状态

counter = 1;
for i = 1:length(x)
    for j = 1:length(xp)
          centers(counter, :) = [x(i), xp(j)];
          counter = counter + 1;
    end
end
dev = [(x(2) - x(1)), (xp(2) - xp(1))] * 1.0; % same deviation + normalized state + diagonal covariance（协方差） -> spherical RBFs  球形rbf
