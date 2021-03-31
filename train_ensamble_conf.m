function [tr_Scores,tst_Scores] = train_ensamble_conf(tr_fv,tr_lbl,tst_fv,tst_lbl)



% make Table
    
%          make Table for training
            fn={};
            for ii=1:size(tr_fv,2)
                fn{ii}=strcat('f',num2str(ii));
            end
            T_tr = array2table(tr_fv,...
            'VariableNames',fn);
            T_tr.lbl = tr_lbl;
            %make testing table 
            
            
            
            
              fn={};
            for ii=1:size(tst_fv,2)
                fn{ii}=strcat('f',num2str(ii));
            end
            T_tst = array2table(tst_fv,...
            'VariableNames',fn);
            T_tst.lbl = tst_lbl;
            
            
            % train base models
             rng('default');
            opt= struct('UseParallel',true,'ShowPlots',false, 'AcquisitionFunctionName','expected-improvement-plus','verbose',0);
           mdls={};
            %svm gaussian
            mdls{size(mdls,2)+1} = fitcsvm(T_tr ,'lbl','KernelFunction','gaussian', ...
             'Standardize',true,'KernelScale','auto');
%              ,'HyperparameterOptimizationOptions',opt,...
%              'HyperparameterOptimizationOptions',opt,...
%             'OptimizeHyperparameters','auto');
%            SVM with polynomial kernel
             rng('default')
             mdls{size(mdls,2)+1} = fitcsvm(T_tr ,'lbl','KernelFunction','polynomial', ...
            'Standardize',true,'KernelScale','auto');
%             ,...
%             'HyperparameterOptimizationOptions',opt,...
%             'OptimizeHyperparameters','auto');
            %liner svm
%              rng('default')
%              mdls{size(mdls,2)+1} = fitcsvm(T_tr ,'lbl','KernelFunction','linear', ...
%             'Standardize',true,'KernelScale','auto');
%             ,...
%             'OptimizeHyperparameters','auto',...
%             'HyperparameterOptimizationOptions',opt);
             %Knn Models
              mdls{size(mdls,2)+1} = fitcknn(T_tr ,'lbl','NumNeighbors',1,'Standardize',1);
              %Knn Models
              mdls{size(mdls,2)+1} = fitcknn(T_tr ,'lbl','NumNeighbors',3,'Standardize',1);
                
              %                     ,...
%                   'OptimizeHyperparameters','auto',...
%                   'HyperparameterOptimizationOptions',opt);
             %Random Forest
              t = templateTree();
               mdls{size(mdls,2)+1}= fitcensemble(T_tr,'lbl');
%                , ...
%                 'OptimizeHyperparameters','auto', ...
%                 'Learners',t, ...
%                 'HyperparameterOptimizationOptions',opt);
            

                %calculate training scores
                N = numel(mdls);
                tr_Scores = zeros(length(tr_lbl),N);
                cv = cvpartition(length(tr_lbl),'kfold',3);
                for ii = 1:N
                    m = crossval(mdls{ii},'cvpartition',cv);
                    [~,s] = kfoldPredict(m);
                    tr_Scores(:,ii) = s(:,m.ClassNames==1);
                end
                tr_Scores;
                
                
                % calculate test scores
                
                tst_Scores = zeros(length(tst_lbl),N);
%                 mdlLoss = zeros(1,numel(mdls));
                for ii = 1:N
                    [lbl,s] = predict(mdls{ii},T_tst);
%                     label = [label,lbl];
                    tst_Scores(:,ii) = s(:,m.ClassNames==1);
%                     mdlLoss(ii) = mdls{ii}.loss(T_tst);
                end
                
                tst_Scores;

end

