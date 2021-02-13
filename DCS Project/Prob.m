function prob = Prob(sig,class)
    prob = zeros(length(class),1);
    for i = 1:length(sig)
        index = find(class == sig(i));
        prob(index) = prob(index) + 1;
    end
    prob = prob/length(sig);
end