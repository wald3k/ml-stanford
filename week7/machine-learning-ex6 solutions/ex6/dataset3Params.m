function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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


function [C, sigma] = findBestCAndSigma()
%This functions runs through all combinations of c and sigma and chooses what
%gives best results on cross validation set.
    c_variants = [0.01; 0.03; 0.1; 0.3; 1; 3 ;10; 30];%This is a row vector
    sigma_variants = [0.01; 0.03; 0.1; 0.3; 1; 3 ;10; 30];%This is a row vector
    score = 9999; %Set some high value, 1 would suffice
    for i=1:size(c_variants)
        for j=1:size(sigma_variants)
                temp_c = c_variants(i);
                temp_sigma = sigma_variants(j);
                model= svmTrain(X, y, temp_c, @(x1, x2) gaussianKernel(x1, x2, temp_sigma)); %Train with X and y
                predictions = svmPredict(model, Xval);
                temp_score = mean(double(predictions ~= yval));
                if temp_score < score
                    printf("For C=%d and sigma=%d --> new min score found: %d .\n", temp_c, temp_sigma, temp_score)
                    score = temp_score;
                    C = temp_c;
                    sigma = temp_sigma;
                endif
        end
    end
end

C = 0.3
sigma = 0.1
%[C, sigma] = findBestCAndSigma() %Run once to get to know what works best (C =0.3 and sigma=0.1)










% =========================================================================

end
