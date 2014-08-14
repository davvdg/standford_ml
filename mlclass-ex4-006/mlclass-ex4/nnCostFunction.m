function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Theta1 has size 25 x 401 25 nodes par 400 dimensions + 1 bias
% Theta2 has size 10 x 26  10 nodes par 26 dimensions + 1 bias 


% k1 = 401 (400 + bias)
% k2 = 26 (25 + bias)
% k3 = 10
             
% Setup some useful variables
m = size(X, 1);
         
X = [ones(m,1) X];  %(m x k1)
z2 = X * Theta1'; %(m x 25)
a2 = [ones(m,1) sigmoid(z2)]; % ( m x k2)
z3 = a2 * Theta2'; %( m x k3)
%a3 = sigmoid(z3);


% You need to retuthe frn ollowing variables correctly 

% X: m x n // theta: k x n
% theta1 

% need to recode y

yr = []; % m x k3
size(y)
yt = y;
for c=1:num_labels
    yr = [yr (yt==c)];
end


%thetaX = X * theta'; %  m x k3
thetaX = z3;
hTheta = sigmoid(thetaX); % m x k3
lht = log(hTheta); % m x k -- log( h(theta X))
lomht = log(1-hTheta); % m x k3
a = -yr .* lht; % m x k3
b = (1-yr).* lomht; % m x k3
c = a - b; % m x k3

reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2,2),1)+sum(sum(Theta2(:,2:end).^2,2),1));




J = (1/m)*sum(sum(c, 2),1) + reg;

del3 = hTheta - yr; % m x k3

gz2 = sigmoidGradient(z2); %(m x 25)

del2 = del3 * Theta2'; % (k2 x k3)' x (m x k3) -> m x k3 x k3 x k2 = m x k2





%J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
