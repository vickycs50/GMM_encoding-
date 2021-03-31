function [pre_result] = near_centroid_classifier(train,test, tr_lbl,tst_lbl)
% data = load('C:\Users\WAQAS\Downloads\wine.data');

% split the dataset to training and testing
% data = data(randperm(end), :);
% train = data(1:floor(0.7*size(data, 1)), :);
% test = data(floor(0.7*size(data, 1))+1:end, :);



% training phase
% --------------------------------------------------------
% initialize the centroid, the first column is the label

centroid = [unique(tr_lbl) zeros(size(unique(tr_lbl), 1), size(train, 2))];


% train = (train - mean(train))/std(train);
% test = (test - mean(test))/std(test);

for label = unique(tr_lbl)'
    % collect all the data of under the label
    train(tr_lbl == label, :);
    % compute the centroid for the label
    centroid(centroid(:, 1) == label, 2:end) = mean(train(tr_lbl == label, :));

end

% testing phase
% --------------------------------------------------------
% initialize the prediction result
pre_result = zeros(size(test, 1), 1);
for i = 1:size(test, 1)
    dist = pdist2(test(i,:), centroid(:, 2:end));
    [~, templabel] = min(dist);
    pre_result(i) = centroid(templabel, 1);

end
end

