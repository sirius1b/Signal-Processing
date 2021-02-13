function [sig_t_q,Ls,Es,vals5] = quantize_dcs(sig_t,L)
% sig_t: time domain discrete signal
% t : corresponding time values
% L : no. of quantization levels
    
    q_max = max(sig_t);
    q_min = min(sig_t);
    delta = (q_max - q_min)/L;
    
    sig_t_q = reshape(sig_t,[length(sig_t),1]);
    vals5 = (q_min : delta: q_max - delta/2) + delta/2;
    vals4 = (q_min : (q_max-q_min)/16: q_max-(q_max-q_min)/32) + (q_max-q_min)/32;
    vals3 = (q_min : (q_max-q_min)/8: q_max -(q_max-q_min)/16 ) + (q_max-q_min)/16;
    vals2 = (q_min : (q_max-q_min)/4: q_max - (q_max-q_min)/8) + (q_max-q_min)/8;
    vals1 = (q_min : (q_max-q_min)/2: q_max -(q_max-q_min)/4) + (q_max-q_min)/4;
    e1 = 0 ; e2 = 0 ; e3 = 0 ; e4 = 0 ; e5 = 0;  
    for i = 1:length(sig_t_q)
        [m5,j5] = min(abs(vals5 - sig_t_q(i))); e5 = e5 + m5^2;
        [m4,j4] = min(abs(vals4 - sig_t_q(i))); e4 = e4 + m4^2; 
        [m3,j3] = min(abs(vals3 - sig_t_q(i))); e3 = e3 + m3^2;
        [m2,j2] = min(abs(vals2 - sig_t_q(i))); e2 = e2 + m2^2;
        [m1,j1] = min(abs(vals1 - sig_t_q(i))); e1 = e1 + m1^2;
        sig_t_q(i) = vals5(j5);
    end
    Es = [e1,e2,e3,e4,e5]/length(sig_t_q);
    Ls = [2,4,8,16,32]; 
end