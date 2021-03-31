function [bagProb] = getBagLabelsFromInstance(test_bags,test_bag_label,testSc,trainSc);

    PL=[];
    bagProb=[];
    start=1;
    en =0;
    for ii=1:size(test_bags)
        s = size(test_bags{ii},1);
        en = en+s;
        prob = testSc(start:en);
        start = start + s;
        
        bagProb(ii)=max(prob);
        
    end
end

