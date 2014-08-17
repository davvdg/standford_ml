function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% X m x n

size(X)
size(theta)
size(y)
a = (X*theta) - y; %(m x 1)
c = theta; % n x 1
c(1,:) = zeros(1,size(c,2)); % ( n x 1, first line is zero)
size(c)

J = (1/(2*m))*(a'*a)+(lambda/(2*m))*(c'*c);
grad = (1/m)* (X'*a) + (lambda/m)*c  % (n x 1)


end
