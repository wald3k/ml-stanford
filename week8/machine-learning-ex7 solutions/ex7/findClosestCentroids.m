function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X,1)
distances = zeros(m, K); %for each train example stores distance to Kth centroid
for i=1:K
    centroids(i)
    dst_centroid_from_train_example = bsxfun(@minus, X, centroids(i,:));%Subtract col vector centroids(i) from X
    dst_centroid_from_train_example = sum(dst_centroid_from_train_example.^2, 2);
    distances(:,i) = dst_centroid_from_train_example;
end
[min_val, idx] = min(distances, [], 2);

%for i=1:size(X)(1)
%    x = X(i);
%    temp = centroids - x;
%    temp = temp.^2;
%    temp = sum(temp, 2);
%    temp = temp.^(1/2);
%    min_value = min(temp);
%    idx(i) = find(temp == min_value);
%end





% =============================================================

end

