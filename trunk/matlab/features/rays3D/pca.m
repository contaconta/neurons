% --- function pca
% PCA analysis of data set.
%function [Y] = pca (X,d)

%clear all
%close all

% n = 1000;
% x = randn(n,1)*100;
% y = randn(n,1)*50;
% z = randn(n,1)*10;
% d = 2;
% 
% X = [x y z];

%X = gsamp([0 0 0], [.7 .1 .3; .1 .3 0; .3 0 .2], 1000);
X = gsamp([0 0 0], [.1 .1 .3; .1 .8 0; .3 0 .3], 1000);

figure;
plot3(X(:,1),X(:,2),X(:,3),'r.');

%[Y,D] = eigs(X*X',d,'lm');
[Y,D] = eigs(X'*X);
Y = Y';

hold on;
h=plot3([0 Y(1,1)*1],[0 Y(2,1)*1],[0 Y(3,1)*1]);
set(h, 'LineWidth', 3);
h=plot3([0 Y(1,2)*1],[0 Y(2,2)*1],[0 Y(3,2)*1],'g');
set(h, 'LineWidth', 3);
h=plot3([0 Y(1,3)*1],[0 Y(2,3)*1],[0 Y(3,3)*1],'m');
set(h, 'LineWidth', 3);

h=plot3([0 2],[0 0],[0 0],'b--');
h=plot3([0 0],[0 2],[0 0],'g--');
h=plot3([0 0],[0 0],[0 2],'m--');

axis equal
grid on
xlabel('x')
ylabel('y')
zlabel('z')


load Mannequin
p3 = Y*p'; p3 = p3';

figure;
plot3(p(:,1), p(:,2), p(:,3), 'b.');
hold on;
axis equal;
plot3(p3(:,1), p3(:,2), p3(:,3), 'g.');
xlabel('x')
ylabel('y')
zlabel('z')
pall = [p; p3];
axis([ min(pall(:,1)) max(pall(:,1))   min(pall(:,2)) max(pall(:,2))  min(pall(:,3)) max(pall(:,3)) ])
grid on