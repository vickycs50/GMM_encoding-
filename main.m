% The default parameters of miVLAD.
clear 
warning('off','all')
addpath('../data/figure');
addpath('../data/musk');
%My_MUSK2.mat
dataset_name = "";
% dataset_name = 'E:\\MIL code\\columbia_dataset\\musk2norm_matlab';%musk1.mat';;%'figure\\elephant.mat';
dataset_name = 'figure\\fox.mat';
% dataset_name = 'musk_data\\musk1.mat';
load(dataset_name);

num_bag = size(data,1);

str = 'musk_data\\musk2_Index.mat';
cluster={};

% load(str);
%%
num_fold = 1;
num_CV = 10;
models={};
% num_of_models =5;
numClustersSet = 2;
K_cluster_size =3;
% numClustersSet =5;
% cluster_set=5;
prop_to_keep = 0.5 ;
acc = zeros(num_fold,num_CV);
acc2 = zeros(num_fold,num_CV);
all_centers={};
all_assignments={};
fi=1;
r=3;%also try 400;
% for mr=1:10
best_ind=[];
best_acc=0;
all_ind=[];
all={};
val_proj=[]
tst_proj=[];
auc=[];
fi=1;
all2={};
nn=[6,17,38,46,97,98];
temp_set=[];

for pca = [1]
for kn = [1]%[6,5,4,3,2,1]
for i=1:10%-1:1%:10%[2,3,4,5,6,7,8,9,1]%:10%10:-1:1%:num_fold


 
    for j =1:10%1:10%[9,8]%1:10%[1,5,6,8]%1:10%-1:1%:num_CV%[7,8,9,10,3,1,2,4,5,6]%1:num_CV %[2,8,9]
     
        tic
 
%         load('musk_data//musk1_Index.mat');
        load('figure//figure_testIndex.mat')
          cur_testIndex = testIndex((i-1)*num_CV+j,:);
        cur_trainIndex = 1:num_bag;
        cur_trainIndex(cur_testIndex) = [];
        num_train_bag = size(cur_trainIndex,2);
        num_test_bag = size(cur_testIndex,2);

           


        num_train_bag = size(cur_trainIndex,2);
        num_test_bag = size(cur_testIndex,2);
%         opt_fv=opt;
       
       
           train_bags = data(cur_trainIndex,1);
           train_bag_labels = data (cur_trainIndex,2);
           test_bags = data(cur_testIndex,1);
           test_bag_labels = data(cur_testIndex , 2);        
            space =zeros(1,size(data{1,1},2));
            train_instances =[];
                 train_lable=[];
                 
                 %extract instances for clusstering
                 Xtb=[];
                 num_train_bag = size(train_bags,1);
                  for ii = 1:num_train_bag%num_train_bag%num_bag%num_train_bag
            %             train_bags_fs{ii,1};
                        train_instances = [train_instances; train_bags{ii,1}]; %#ok<AGROW>
                        for jj=1:size(train_bags{ii,1},1)
                         
                          train_lable = [train_lable; train_bag_labels{ii}];
                          Xtb = [Xtb;cur_trainIndex(ii)];
                        end

                  end
                  D.X=train_instances;
                  D.Y= train_lable;
                  D.B = cur_trainIndex';
                  D.YB = cell2mat(train_bag_labels);
                  D.XtB = Xtb;
                  D.YR = train_lable;
%                   %musk2
                  nSel =2;
                  nK = 5;    % number of clusters in k-means
                  nRS =15;   % number of random subspaces
                  nDSS = 0.3;   % number of dimension per subspace
                  T = 0.0100;     % temperature for softmax selection
                  
                  
                  nSel =2;
                  nK = 10;    % number of clusters in k-means
                  nRS =10;   % number of random subspaces
                  nDSS = 0.;   % number of dimension per subspace
                  T = 0.0100;     % temperature for softmax selection
                  
                  
                  
                  
                   nSel =2;
                  nK = 5;    % number of clusters in k-means
                  nRS =10;   % number of random subspaces
                  nDSS = 0.05;   % number of dimension per subspace
                  T = 0.01;%250;    
                  
                  

                res = zeros(nRS,size(D.X,1));

                nDSS = ceil(size(D.X,2)*nDSS);
                pop=[];
                RSS_detail={};
                ens_label=[];
          ens_yval=[];
           ens_label2=[];
          ens_yval2=[];
            for kk=1:5
                for s = 1:nRS
                        % Randomly select the dimension for the subspace
                        ind = randperm(size(D.X,2));
                        ind = ind(1:nDSS);
                        R = false(size(D.X,2), 1);
                        R(ind) = true;
                         % Cluster data in the subspace
%                         [C, A] = vl_kmeans(D.X(:,R)', nK);
%                          C = C';
%                          RSS_detail{s,1}=C;
%                          RSS_detail{s,2}=A;
%                          RSS_detail{s,3}=R;
%                          % compute proporition of positive bag per cluster
%                         pC = zeros(size(C,1),1);
%                         ctnC = zeros(size(C,1),1);

                         pos_prob=[];
                         pos_sc=0;
                         
%                          options = [1.5 NaN NaN 0];
%                         [centers,U] = fcm(D.X(:,R),nK,options);
%                          C = centers;
%                          C=centers;

%                          U_prob = sum(U(:,1:max(find(D.Y==1))),2)./sum(U,2);
%                          U_prob2 = sum(U(:,max(find(D.Y==1))+1:end),2)./sum(U,2);
%                          prob = [];
                         
                         
                         Sigma = {'diagonal','false'}; % Options for covariance matrix type
                            nSigma = numel(Sigma);

                            SharedCovariance = {true,false}; % Indicator for identical or nonidentical covariance matrices
                            SCtext = {'true'};
                            nSC = numel(SharedCovariance);
                          all_prob={};
                          fi_p=1;
                          options = statset('MaxIter',20);
                         for ix = 1:nSigma
                             
                            [posteriors]= vl_gmm_examples(D.X(:,R)',nK,ix);
                            posteriors;
                            all_prob{1,fi_p}=posteriors';
                            P = posteriors';
                            fi_p = fi_p+1;
% %                             for jx = 1:nSC
%                                  gmfit = fitgmdist(D.X(:,R),nK,'CovarianceType',Sigma{ix},'SharedCovariance' ,true,'Options',options);%, ...
%                                  %'SharedCovariance',SharedCovariance{jx})%, 'Options',options);
%                                   threshold = [0.4 0.6];
%                                   P = posterior(gmfit,D.X(:,R));
%                                   all_prob{1,fi_p}=P;
%                                   fi_p = fi_p+1;
%                                   
% %                             end
                         end
                         probs=zeros(size(P,1),nK);
                         for ink=1:nK
                             ss=zeros(size(P,1),size(all_prob,2));
                             for jpi=1:size(all_prob,2)
                                 P=all_prob{1,jpi};
                                 ss(:,jpi) = P(:,ink);
                                 
                             end
                             probs(:,ink)=mean(ss,2);
                             
                         end
% vl_gmm_examples(D.X(:,R)',nK)
%                         [means,covariances,priors,ll,posteriors]=vl_gmm(D.X(:,R)',nK)
                         U = probs';
                         U_prob = sum(U(:,1:max(find(D.Y==1))),2)./sum(U,2);
                         U_prob2 = sum(U(:,max(find(D.Y==1))+1:end),2)./sum(U,2);
                         prob = [];
                         
                         

                         for cc=1:size(U,2)
                             prob(cc,1)=sum(U(:,cc).*U_prob);
                             
                             
                         end
                        res(s,:) = prob';
                end
                score = mean(res,1);
                probX = score*0;
                %% create softmax probability vector
                bagList = unique(D.XtB);
                for ii = 1:length(bagList)

                    % get denuminator
                    idb = (D.XtB == bagList(ii));
                    denum = sum(exp(score(idb)/T));

                    % get index of elements in the bag
                    idi = 1:length(D.XtB);
                    idi = idi(idb);

                    for jj = 1:length(idi)
                        probX(idi(jj)) = exp(score(idi(jj))/T)/denum;
                    end

                end
               
             selection = false(nSel,size(D.X,1));
                for jj = 1:nSel

                        % select one instance per bag
                        for ii = 1:length(bagList)
%                            
                            % get probabilities of instance of the bag
                            idb = (D.XtB == bagList(ii));
                            p = probX(idb);

                            % get index of elements in the bag
                            idi = 1:length(D.XtB);
                            idi = idi(idb);

                            % get index of the selected instance
                           
                            cumm = 0;
                             ind=[];

                            [B,I] = sort(p);
%                             p = p(I);
%                             idi = idi(I);
                            t=rand();
                            for k = 1:length(idi)
                                cumm = cumm + p(k);
%                                 ind = k;
                                ind = [ind;k];
                                if t < cumm
                                    break;
                                end
                            end                          
                            selection(jj,idi(ind)) = true;

                        end
                end  
                pop=selection;
         
         

          models={};         
          n_train_bags = {};         
          for kkp=1:size(pop,1)
             for ii = 1:num_train_bag
                      bag = train_bags{ii};
                     
%                        for jj = 1:length(bagList)

                            % get probabilities of instance of the bag
                            idb = (D.XtB == bagList(ii));
%                             p = probX(idb);
                             % get index of elements in the bag
                            idi = 1:length(D.XtB);
                            idi = idi(idb);
%                             ind = selection(idi,:);
                            ind = pop(kkp,idi);
                            bag = bag(logical(ind),:);
                            n_train_bags{ii,1}=bag;
                           
                           
           
             end
%              
             RSS_detail;
             n_test_bags = test_bags;


%                             n_test_bags = test_bags;
                            opt.kmeans_num_center =kn;%5;%1;%best_ksize;%1;
                            opt.PCA_energy = pca;%0.95;%0.85;%best_pca; % 0.95;
                            temp_opt=opt;
                            if pca==1
                                 opt.PCA_energy=0;
                            end
%                             opt.R =randi([0 1],1,size(data{1,1},2) );
%                             opt.R =logical(opt.R);
                        nDFV=1;
                         nDFV = ceil(size(D.X,2)*nDFV);
                        ind = randperm(size(D.X,2));
                        ind = ind(1:nDFV);
                        R = false(size(D.X,2), 1);
                        R(ind) = true;
                         opt.R = R;
%                             [tr_fv,tst_fv,val_fv] = getFV_validation(n_train_bags,train_bag_labels,n_test_bags,test_bag_labels,train_bags,opt);
                            R;
                           
                            type = "fv";
                           
                            [tr_fv,tst_fv,op,codes]=getFV(n_train_bags,train_bag_labels,n_test_bags,test_bag_labels,opt,[],[],[],type);
                           
                             [val_fv,t_fv,~,~]=getFV(train_bags,train_bag_labels,n_test_bags,test_bag_labels,opt,[],op,codes,type);
                           
%                             [tr_fv,tst_fv,op,codes]=getFV(n_train_bags,train_bag_labels,n_test_bags,test_bag_labels,opt,[],[],[]);
%                            
%                              [val_fv,t_fv,~,~]=getFV(train_bags,train_bag_labels,n_test_bags,test_bag_labels,opt,[],op,codes);
                          
                             [tr_score,tst_score] = train_ensamble_conf(tr_fv, cell2mat(train_bag_labels), tst_fv, cell2mat(test_bag_labels));
                            ens_label = [ens_label,tst_score];
                            ens_yval = [ens_yval, tr_score];
                            
                                n_train_bags={};
                                                  
                           
          end
            end
           ens_label1 =[ ens_label];
           ens_yval1 = [ens_yval];
          sc=mean(ens_label1,2);          
          sc2 = mean(ens_yval1,2);
          cheat_t = findThreshold(ens_label,cell2mat(test_bag_labels));
          pl=sc>cheat_t;
         
          cp = classperf(logical(cell2mat(test_bag_labels)),pl);
          cheat_acc(i,j)=1-cp.ErrorRate;
         
          addpath('svm//')
         
          ens_yval2 = zscore(ens_yval1);
          ens_label2 = zscore(ens_label1);

                minv = min(ens_yval2);
                maxv = max(ens_yval2) - minv;
                maxv = maxv +eps;
                maxv = 1./maxv;
                ens_yval2 = (ens_yval2 -repmat(minv,num_train_bag,1)) .* repmat(maxv,num_train_bag,1);
          
       
% %                  minv = min(ens_label2);
% %                 maxv = max(ens_label2) - minv;
% %                 maxv = maxv +eps;
% %                 maxv = 1./maxv;
          ens_label2 = (ens_label2 -repmat(minv,num_test_bag,1)) .* repmat(maxv,num_test_bag,1);
         model = train(cell2mat(train_bag_labels),sparse(ens_yval2),'-s 1 -c 0.05 -B -1 -q');
        [pred_t, accuracy, dec_val] = predict(cell2mat(test_bag_labels),sparse(ens_label2),model);
       svm_acc(i,j)=accuracy(1);
         rmpath('svm//');
         
         
         learner = templateKNN('NumNeighbors',3); 
          
          ens = fitcensemble(ens_yval2,cell2mat(train_bag_labels),'Method','Subspace','NumLearningCycles',200,...
          'Learners',learner,'NPredToSample',5);
          pl3=predict(ens,ens_label2);
          
         
          C = confusionmat(logical(cell2mat(test_bag_labels)),logical(pl3));
          accuracy = sum(diag(C))/size(test_bag_labels,1);
         
         
         
         
         
          voting_acc2(i,j)=accuracy(1);
          voting_acc2;
       
       [pre_result] = near_centroid_classifier(ens_yval2,ens_label2, cell2mat(train_bag_labels),cell2mat(test_bag_labels));
          C = confusionmat(logical(cell2mat(test_bag_labels)),logical(pre_result));
          accuracy = sum(diag(C))/size(test_bag_labels,1);
       cent_acc(i,j) = accuracy(1);
       
          train_project = [];
          test_project = [];
          val_proj=[];
          tst_proj=[];
          [X,Y,T,AUC] = perfcurve(cell2mat(test_bag_labels),sc,1);
          auc(i,j)=AUC;
          
          save('confM2');
          
          perfObjB_svm{i,j} = getClassifierPerfomance(pred_t,cell2mat(test_bag_labels),sc);
          perfObjB_voting{i,j} = getClassifierPerfomance(pl3,cell2mat(test_bag_labels),sc);
          perfObjB_ncen{i,j} = getClassifierPerfomance(pre_result,cell2mat(test_bag_labels),sc);
 
    end
    
   

end

   all2{fi,1}=pca;
   all2{fi,2}=kn;
   all2{fi,3}=mean(mean(voting_acc2));
   all2{fi,4}=mean(mean(cent_acc));
   all2{fi,5}=mean(mean(svm_acc));
   all2{fi,6}=mean(mean(auc));
   all2{fi,7}=voting_acc2;
   all2{fi,8}=cent_acc;
   all2{fi,9}=svm_acc;
   all2{fi,10}=auc;
  
   perfB_ncen = getMeanPref(perfObjB_ncen );
   perfB_svm = getMeanPref(perfObjB_svm );
   perfB_vote = getMeanPref(perfObjB_voting );
   
   all2{fi,11}=perfB_ncen;
   all2{fi,12}=perfB_svm;
   all2{fi,13}=perfB_vote; 
   all2{fi,14}=perfObjB_ncen;
   all2{fi,15}=perfObjB_svm;
   all2{fi,16}=perfObjB_voting;
   
     
   
   acc=[];
   voting_acc2=[];
   svm_acc=[];
   cent_acc=[];
   
   fi=fi+1;
   
   save('confM1');

end
end

mean(acc,2)
disp(['Accuracy = ',num2str(mean(mean(acc2))),'¡À',num2str(std(acc2(:)))]);
acc;



