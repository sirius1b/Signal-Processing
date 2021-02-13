clf;
clc;
clear;
%% 
addpath('src/');
[sig_t,sig_f,t,Fs,L,sig_f_all] = input('src/two.wav');
% [sig_t,sig_f,t,Fs,L,sig_f_all] = input('src/parrot.txt');
% [sig_t,sig_f,t,Fs,L,sig_f_all] = input('sd') ;

[s1_t,t1,s2_t,t2] = source(sig_t,sig_f,t,Fs,L);
[s1_tq,Ls1,Es1,levels1] = quantize_dcs(s1_t,32);
[s2_tq,Ls2,Es2,levels2] = quantize_dcs(s2_t,32);

subplot(3,3,5);
plot(t1,s1_t);
title("Upsampled Signal")
subplot(3,3,6);
plot(t2,s2_t);
title("Downsampled Signal")
subplot(3,3,7);
plot(Ls1,Es1);
title("Variation of MSE of Upsampled Signal")
subplot(3,3,8);
plot(Ls2,Es2);
title("Variation of MSE of Downsampled Signal")

bits1U = encode_dcs(s1_tq,levels1,[],0);
bits2U = encode_dcs(s2_tq,levels2,[],0);
prob1 = Prob(s1_tq,levels1);
prob2 = Prob(s2_tq,levels2);

levels1 = reshape(levels1,size(prob1));
levels2 = reshape(levels2,size(prob2));
code1 = huffman_dcs(levels1,prob1);
code2 = huffman_dcs(levels2,prob2);
bits1H = encode_dcs(s1_tq,levels1,code1,1);
bits2H = encode_dcs(s2_tq,levels2,code2,1);

[m,tm,rcP1,trc] = PulseShaping(bits1H,(1/(5*Fs))*(10^5));

subplot(3,3,9);

plot(tm,m);
title("Part 7")
xlabel("t[n]")
ylabel("")


f3 = figure;
subplot(1,1,1);
plot(trc,rcP1);

f2 = figure ; 
figure(f2);
subplot(2,1,1)
plot(t1,s1_tq);
title("Quantized Upsampled Signal")
subplot(2,1,2)
plot(t2,s2_tq);
title("Quantized Downsampled Signal")




%% function
function [sig_t,sig_f,t,Fs,L,sig_f_all] = input(m)
    if contains(m,".wav")
        info = audioinfo(m);
        [y,Fs] = audioread(m);
        t = 0:seconds(1/Fs):seconds(info.Duration);
        t = t(1:end-1);
        sig = y;
%         plot(t,y)
%         xlabel('Time')
%         ylabel('Audio Signal')
        
       % fft: 0 at last , last in middle
        
    elseif contains(m,".txt")
        fid = fopen(m);
        x = fread(fid,'*char');
        binary = dec2bin(x,8);
        b_t = transpose(binary);
        bin = b_t(:) - '0';
        t = 1:length(bin);
        sig = bin;
        %plot(t,bin);
        Fs = 1; % 1 character per sample
    else
        Fs = 100;
        T = 1/Fs;
        L = 2000; 
        t = (0:L-1)*T;
        sig = 2*sin(2*pi*10*t) + 3*sin(2*pi*t)+0.5*cos(2*pi*30*t);    
    end
    
    y2 = fft(sig);
    y1 = y2(1:length(y2)/2);
    f = Fs*(0:length(y2)/2-1)/length(y2);
    
    sig_f_all = y2;
    sig_t = sig;
    sig_f = y1;
    L = length(t);
    
    
    
    subplot(3,3,1);
    plot(t,real(sig));
    
    title("Amplitude Plot(Time domain)");
    ylabel("Amplitude")
    xlabel("time(t)")
    
    subplot(3,3,2);
    plot(t,angle(sig));
    title("Phase Plot(Time domain)");
    ylabel("Phase")
    xlabel("time(t)")

    subplot(3,3,3);
    plot(f,abs(y1./L));
    title("Single Sided Amplitude Spectrum");
    ylabel("Amplitude")
    xlabel("Freq(f)")
    
    subplot(3,3,4);
    plot(f,angle(2*pi.*y1./L));
    title("Single Sided Phase Spectrum");
    ylabel("Phase")
    xlabel("Freq(f)")
end




