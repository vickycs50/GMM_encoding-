
function [posteriors] = vl_gmm_examples(data,numClusters,t)
% Run KMeans to pre-cluster the data
% numClusters = 30;
numData = size(data,2);
dimension = size(data,1);
% data = rand(dimension,numData);
[initMeans, assignments] = vl_kmeans(data, numClusters, ...
    'Algorithm','Lloyd', ...
    'MaxNumIterations',5);

initCovariances = zeros(dimension,numClusters);
initPriors = zeros(1,numClusters);

% Find the initial means, covariances and priors
for i=1:numClusters
    data_k = data(:,assignments==i);
    initPriors(i) = size(data_k,2) / numClusters;
   if t==1
        if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = diag(cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
   else
       if size(data_k,1) == 0 || size(data_k,2) == 0
            initCovariances(:,i) = (cov(data'));
        else
            initCovariances(:,i) = diag(cov(data_k'));
        end
   end
end

% Run EM starting from the given parameters
[means,covariances,priors,ll,posteriors] = vl_gmm(data, numClusters, ...
    'initialization','custom', ...
    'InitMeans',initMeans, ...
    'InitCovariances',initCovariances, ...
    'InitPriors',initPriors);
end