function [code] =  huffman_dcs(class,probs)
    p = probs ;
    code = strings(size(class));
    prob_child = [p,string(reshape(1:length(class),size(p)))]; % probabilities and childs of each node
    allowed = ones(size(class));
    while (sum(allowed == 1) > 1)
        pc_sorted = sortrows(prob_child,1);
        min1= pc_sorted(1,1);
        min2 = pc_sorted(2,1);
        c1 = pc_sorted(1,2); c2 = pc_sorted(2,2);
        new_childs = "";
        for i = str2num(char(split(c1,',')))'
            code(i) = 1 + code(i);
            new_childs = new_childs + "," + i;
            allowed(i) = 0;
        end
        for i = str2num(char(split(c2,',')))'
            code(i) = 0 + code(i);
            new_childs = new_childs + "," + i ;
            allowed(i) = 0;
        end
        p3 = str2double(min1)+str2double(min2); 
        index1 = find(prob_child(:,2) == c1);
        allowed(index1) = 0;
        prob_child(index1 ,1) = 2;
        index2 = find(prob_child(:,2) == c2);
        allowed(index2) = 0;
        prob_child(index2, 1) = 2;
        pp = [string(p3),string(new_childs)];
        prob_child = [prob_child;pp];
        allowed = [allowed;1];    
    end
    
end