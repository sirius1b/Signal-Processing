
%*************** Part 1
x = 5;
y = 2;
u  = x + y;
v = x*y ;
w = x/y;
z = w^3;
s = x*y^2/(x-y) ;
p = 3*x/(2*y);
r = 3*x*y/2 ; 
t = x^5/(x^5-1);

%**************** Part 2
x = 10 ; 
y = 3;
r = 8*sin(y);
s = 5*sin(2*y);
z = sin(x);
w = 2*sin(x);
p = exp(x-1);
u = 2+cos(2*pi*x);
m = sqrt(x) + 4 + sin(0.2*pi) + exp(2);

%**************** Part 3
x= 3, y = 4;
func1(x,y);
func2(x);
func3(x,y);

x = [3; 1; 0];
y = [0;1;1];
func1(x,y);
func2(x);
func3(x,y);

x = [- 3  1 0 ; 1 0 1 ] ;
y = [1 1 1 ; 2 0 -2];
func1(x,y);
func2(x);
func3(x,y);
%************************ Part 4 
A = [1:7 ; 9:-2:-3 ; 2.^(2:8)]

%*********************** Part 5
hold on;
f = @(x) x./(x + 1./(x.^2))
x = 0:0.01:7
y = f(x)
plot(x, y);
%********************** Part 6
x = 0:0.001:2*pi ;
y = sin(x.^2); 
plot(x,y);
xlabel("X-Axis")
ylabel("Y-Axis")
title("Plots")
legend("Ques 5", "Ques 6")
grid on;



% ********************* functions
function op = func1(x,y)
	op = 3/2.*x.*y
end

function op = func2(x)
	op = (1- 1./x).^-1
end 

function op = func3(x,y)
	op = 4.*(y-5)./(3.*x-6)
end
