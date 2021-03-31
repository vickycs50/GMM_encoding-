function [concepts_set] = get_noc(concepts,relv, noc, class)
          
        temp_pos_concepts = concepts;
        temp_relv = relv;
        concepts_set={};
        
        %select most relevant concept
        [val,ind] = max(relv);
        concepts_set{1,1} = cell2mat(temp_pos_concepts(ind,1));
%         candiate_set{1,2} =temp_relv(ind);
%         candiate_set{1,3} =1;
        temp_pos_concepts(ind,:)=[];
        temp_relv(ind)=[];


      for jj=1:noc
            % compute redundency from concepts
            diversity = [];
            for ii=1:size(temp_pos_concepts,1)

                A = temp_pos_concepts{ii,1};
                dif=[];
                for cs=1:size(concepts_set,1)
                    
                    B = concepts_set{cs,1};
                    di = setdiff( A, B );
                    dif = [dif; size(di,1)];
                end
                if isinf(min(dif)/length(B(:)))
                    diversity;
                end
                diversity(ii,1)= min(dif)/length(B(:));
            end
            
            dif
            
            
            
            
            % r is avergae relavance
            % d is average diversity
            r = mean(temp_relv);
            d = mean(diversity);
            
            %select more ralevant and diverse concept
            % find concepts > average ralevance concepts
             sel_ind=find(temp_relv>r & diversity > d);
                     
             [val, ind]=max( mean([temp_relv(sel_ind), diversity(sel_ind)],2));
             concepts_set{jj,1} = cell2mat(temp_pos_concepts(sel_ind(ind),1));
             temp_pos_concepts(ind,:)=[];
             temp_relv(ind)=[];
             
      end
         
end

