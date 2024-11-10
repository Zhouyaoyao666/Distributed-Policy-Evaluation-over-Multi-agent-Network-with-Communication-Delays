function [ phi ] = phi( s,centroids,dev)
% computes phi(s), covariance matrix assumed to be diagonal -> each
% gaussian can be evaluated separately and then multiplied
%假定协方差矩阵为对角阵，每个高斯单独计算然后再相乘

phi = zeros(16, 1);
% rng(234)% 生成相同的随机数
% H =25*rand(27,3);
% d = diag([1,1,0.2]);

for i = 1:16
      phi(i)=exp(9*((s(1)-centroids(i, 1))^2+(s(2)-centroids(i, 2))^2));
end
%       phi=0.5*cos(H*d*s');

% phi = zeros(length(centroids), 1);
% for i = 1:length(centroids)
%     rbf1 = exp(-4*abs(centroids(i, 1) - s(1)) / (2 * dev(1)^2));
%     rbf2 = exp(4*abs(centroids(i, 2) - s(2)) / (2 * dev(2)^2));
%     phi(i) = rbf1 * rbf2;
% end


phi = phi/sum(phi); % normalize output 归一化