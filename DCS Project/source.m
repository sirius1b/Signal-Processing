function [s1_t,t1,s2_t,t2] =  source(sig_t,sig_f,t,Fs,L)
    sf = abs(sig_f) ~=  0;
    max_index = find(sf,length(sf),'last');
    max_index = max(max_index);    
    % upsample 
    s1 = sig_f;
    s1(max_index+1:round(max_index*4))  = 0;
    s1_t = real(ifft(s1));
    t1 = (0:length(s1_t)-1)/Fs;
    
    % downsample
    s2 = sig_f;
    s2(round(max_index*0.75):max_index) = 0;
    s2 = s2(1:max_index);
    s2_t = real(ifft(s2));
    t2 = (0:length(s2_t)-1)/Fs;    
end