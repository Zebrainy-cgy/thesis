% 创建一个 x 和 y 的范围  
x = 0:0.05:1;  
y = 0:0.05:1;  
  
[X, Y] = meshgrid(x, y);  
P = exp(-X.^2-Y.^2); 
  
% 绘制图像  
grid off;
surf(X, Y, P);  
xlabel('x');  
ylabel('y');  
zlabel('p');  
title('Graph of p = exp(-x^2-y^2)');  