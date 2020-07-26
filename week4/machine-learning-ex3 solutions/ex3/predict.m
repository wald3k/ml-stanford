function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%



for i = 1:m
    %Calculations for the second (hidden) layer
    A1 = X(i,:);
    A1 = [ones(size(A1), 1) A1];
    Z2 = A1 * Theta1';
    A2 = sigmoid(Z2); %Size (25 x 401)   
    A2 = [ones(size(A2,1), 1) A2];
    
    %Calculations for the third (output) layer
    Z3 = A2 * Theta2';
    A3 = sigmoid(Z3); %Size (401 x 10);

    % Assigning label of the most probable score
    [val, ix] = max(A3, [], 2);
    p(i) = ix;

end



% =========================================================================


end
