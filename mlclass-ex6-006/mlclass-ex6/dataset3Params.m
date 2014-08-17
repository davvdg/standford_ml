function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

ret =zeros(size(values,1)^2,2);
i = 1;
error = zeros(size(values,1)^2,1);
for cv = 1:size(values)
    for sigmas = 1:size(values)
        ret(i,:) = [values(cv), values(sigmas)];
        ret(i)
        model= svmTrain(X, y, values(cv), @(x1, x2) gaussianKernel(x1, x2, values(sigmas)));
        predictions = svmPredict(model, Xval);
        error(i) = mean(double(predictions ~= yval));
        error(i)
        i = i+1;
    end
end
[mi, idx] = min(error);
idx

ret
error

C = ret(idx, 1);
sigma = ret(idx,2);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
