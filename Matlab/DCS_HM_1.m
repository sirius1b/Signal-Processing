%Author: Lavanya Verma
% DCS Monsoon 2020, First Non-graded Assignment
% Submission Date : August 28, 2020
%Comment on nature of plots:
% With increase in n(size of superalphabet), no. of bits required to encode
% each symbol increases. However, plot of functions shows that as n
% increases, rate of transmission(L) converges to some finite value, which
% is heavy depended upon M.
% 
% f  : converging nature 
% f1 : appears to be constant
% f2 : increasing but non differentiable due to nature of ceil
% f3 : strictly increasing 
% *****************Function Definitions*************************
f = @(M,n) ceil(log2(M.^n))./n;
f1 = @(M,n) log2(M.^n)./n;
f2 = @(M,n) ceil(log2(M.^n));
f3 = @(M,n) log2(M.^n);
%************************************************


%*************Uncomment only line********************
n  = linspace(1,50,10000);
% n = linspace(1,20,1000);
%************************************

subplot(2,2,1);
title("Plot of ceil(log2(M^n))/n vs n")
xlabel("n->")
y1 = f(3 ,n );
y2 = f(4, n);
y3 = f(5, n);
y4 = f(8 , n);
y5 = f(16 , n);
y6 = f(25 , n);
hold on
plot(n,y1);
plot(n,y2);
plot(n, y3); 
plot(n, y4);
plot(n, y5); 
plot(n, y6);
legend("M = 3", "M = 4","M = 5","M =8","M = 16","M = 25" );

subplot(2,2,2);
title("Plot of log2(M^n)/n vs n")
xlabel("n->")
y1 = f1(3 ,n );
y2 = f1(4, n);
y3 = f1(5, n);
y4 = f1(8 , n);
y5 = f1(16 , n);
y6 = f1(25 , n);
hold on
plot(n,y1);
plot(n,y2);
plot(n, y3); 
plot(n, y4);
plot(n, y5); 
plot(n, y6);
legend("M = 3", "M = 4","M = 5","M =8","M = 16","M = 25" );

subplot(2,2,3);
title("Plot of ceil(log2(M^n))")
xlabel("n->")
y1 = f2(3 ,n );
y2 = f2(4, n);
y3 = f2(5, n);
y4 = f2(8 , n);
y5 = f2(16 , n);
y6 = f2(25 , n);
hold on
plot(n,y1);
plot(n,y2);
plot(n, y3); 
plot(n, y4);
plot(n, y5); 
plot(n, y6);
legend("M = 3", "M = 4","M = 5","M =8","M = 16","M = 25" );

subplot(2,2,4);
title("Plot of log2(M^n))")
xlabel("n->")
y1 = f3(3 ,n );
y2 = f3(4, n);
y3 = f3(5, n);
y4 = f3(8 , n);
y5 = f3(16 , n);
y6 = f3(25 , n);
hold on
plot(n,y1);
plot(n,y2);
plot(n, y3); 
plot(n, y4);
plot(n, y5); 
plot(n, y6);
legend("M = 3", "M = 4","M = 5","M =8","M = 16","M = 25" );



