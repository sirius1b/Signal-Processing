function bits =  encode(sig,class,code,flag) 
%     flag: 1-> huffman, 0-> normal binary
        bits = [];
        for i = 1:length(sig)
            if flag == 0 
                cd = de2bi(find(class == sig(i)) - 1,ceil(log2(length(class))));
                bits = [bits,cd];
            elseif flag == 1
                cd = code(find(class == sig(i)));
                b = split(cd,'');
                b = b(2:end-1);
                bits = [bits, b'];
            end    
        end
    if (flag == 0)
        bits = bits';
    elseif (flag == 1)
        bits = str2num(char(bits'));
    end
end