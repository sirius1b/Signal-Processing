%%fourier fft, dft
% Fs = 100;
% T = 1/Fs;
% L =10000;
% t = (0:L-1)*T;
% S = sin(2*pi*30*t) + 3*cos(2*pi*40*t);
% Sf1 = fft(S);
% Sf2 = dft(reshape(S,[L,1]));
% 
% f = Fs*(0:L/2-1)/L;
% subplot(2,1,1)
% sf1 = abs(Sf1(1:L/2));
% plot(f,sf1);
% subplot(2,1,2);
% sf2 = abs(Sf2(1:L/2));
% plot(f,sf2);

