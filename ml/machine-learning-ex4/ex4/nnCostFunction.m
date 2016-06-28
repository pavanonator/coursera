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

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
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


% Add ones to the X data matrix
X = [ones(m, 1) X];
%create y matrix which has y vector for each class
y2=y;
y3=y;
for i=2:num_labels
  y=[y y2==i];
  y2=y3;
endfor
y(:,1)=(y3==1)';
size(y);
%vectorized forward prop (w/o regularization)
a2 = sigmoid(X*Theta1');
a2=[ones(m,1) a2]; %bias unit addition
h_of_x = sigmoid(a2*Theta2');
J = 1 / m * sum( -1 .* y(:) .* log(h_of_x(:)) - (1-y(:)) .* log(1 - h_of_x(:)) ); 

%now regularize Theta1 and 2 disregarding bias units
zeros1=zeros(hidden_layer_size,1);
Theta1_reg = [zeros1 Theta1(:,2:(input_layer_size+1))];
zeros2=zeros(num_labels,1);
Theta2_reg = [zeros2 Theta2(:,2:(hidden_layer_size+1))];
J += lambda / (2 * m) * (sum( sum(Theta1_reg .^ 2)) + sum( sum(Theta2_reg .^ 2 )));


% -----------------------------do the grads--------------------------------
y2=y3; %original y outputs 5000x1
d3=ones(num_labels,1);
D=0;
for i=1:m
  y3=y2;
  a1=X(i,:)';
  a2=[1; sigmoid(Theta1*a1)]; %26x1
  a3=sigmoid(Theta2*a2); %produces 10x1 output vec
  size(a3)
  for j=1:num_labels
    y=(y3==j);
    d3(j) = a3(j) - y(j);
    y3=y2;
  endfor

  d2 = Theta2(:,2:hidden_layer_size+1)'*d3.*sigmoidGradient(Theta1*a1);
  %d2 = d2(2:end);

  Theta1_grad += (1/m)*(d2*a1');
  Theta2_grad += (1/m)*(d3*a2');
endfor



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
