function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    h = X * theta - y;
    
    %theta0 = theta(1, 1);
    %theta1 = theta(2, 1);
    
    %x0 = v' * X(:, 1);
    %x1 = v' * X(:, 2);
      
    %theta0 = theta0 - alpha * x0 / m;
    %theta1 = theta1 - alpha * x1 / m;
    %theta = [theta0; theta1];
    
    v = X' * h;
    theta = theta - v.* alpha/m;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
