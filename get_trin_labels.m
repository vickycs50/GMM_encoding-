function [ val_grouphat]  = get_trin_labels(tr_fv,train_bag_labels,model)
hpartition = cvpartition(size(tr_fv,1),'Holdout',0.7);
fvtest=tr_fv(hpartition.test);
fvtrain=tr_fv(~hpartition.test);
testlabs=(train_bag_labels(hpartition.test));
trainlabs=(train_bag_labels(~hpartition.test));

% val_group =testlabs;
        cmd=' -c 0.01 -B -1 -q';
%      model = train(trainlabs,fvtrain,cmd);
    [pred_label, accuracy, dec_val] = predict(train_bag_labels,tr_fv,model);
    
    val_grouphat = pred_label;


end

