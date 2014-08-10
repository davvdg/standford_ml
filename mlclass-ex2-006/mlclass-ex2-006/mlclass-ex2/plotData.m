function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
xplus = [];
xminus = [];
for R=1:size(y)
    if (y(R))>.5
        xplus = [xplus; X(R,:)];
    else
        xminus = [xminus; X(R,:)];
    end
end
figure;
hold on;
plot(xplus(:,1),xplus(:,2), 'k+', 'LineWidth', 2);
plot(xminus(:,1),xminus(:,2), 'ko', 'MarkerFaceColor', 'y');

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%









% =========================================================================



hold off;

end
