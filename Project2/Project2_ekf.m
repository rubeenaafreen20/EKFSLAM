% Extended Kalman filter
%
% -------------------------------------------------------------------------
%
% State space model is
% X_k+1 = f_k(X_k) + V_k+1   -->  state update
% Y_k = h_k(X_k) + W_k       -->  measurement
% 
% V_k+1 zero mean uncorrelated gaussian, cov(V_k) = Q_k
% W_k zero mean uncorrelated gaussian, cov(W_k) = R_k
% V_k & W_j are uncorrelated for every k,j
%
% -------------------------------------------------------------------------
%
% Inputs:
% f = f_k
% Q = Q_k+1
% h = h_k
% y = y_k
% R = R_k
% del_f = gradient of f_k
% del_h = gradient of h_k
% x_hat = current state prediction
% P_hat = current error covariance (predicted)
%
% -------------------------------------------------------------------------
%
% Outputs:
% x_next = next state prediction
% P_next = next error covariance (predicted)
% x_dgr = current state estimate
% P_dgr = current estimated error covariance
%
% -------------------------------------------------------------------------
%
%%      Theory
%   px` = px + vx * dt
%   py` = py + vy * dt
%   vx` = vx 
%   vy` = vy 
%   |px`|     | 1  0 dt  0 |   | px |
%   |py`|  =  | 0  1  0 dt | * | py |
%   |vx`|     | 0  0  1  0 |   | vx |
%   |vy`|     | 0  0  0  1 |   | yz |
%    x` = F * x
%%


clc;
clear;
close all;
format shortG;
%% %%%%%%%%%%%%%%%ALL SENSOR DATA AND GROUND TRUTH DATA %%%%%%%%%%%%%%%
filename = 'data_ekf.xlsx';
A = xlsread(filename);
X_lidar=zeros(250,1);
Y_lidar=zeros(250,1);
timestamp_lidar=zeros(250,1);
gt_x_lidar=zeros(250,1);
gt_y_lidar=zeros(250,1);
gt_vx_lidar=zeros(250,1);
gt_vy_lidar=zeros(250,1);
gt_yaw_lidar=zeros(250,1);
gt_yawrate_lidar=zeros(250,1);
rho_radar=zeros(250,1);
pi_radar=zeros(250,1);
Rdot_radar=zeros(250,1);
timestamp_radar=zeros(250,1);
gt_x_radar=zeros(250,1);
gt_y_radar=zeros(250,1);
gt_vx_radar=zeros(250,1);
gt_vy_radar=zeros(250,1);
gt_yaw_radar=zeros(250,1);
gt_yawrate_radar=zeros(250,1);
k=0;
for i=1:2:499
    k=k+1;
    X_lidar(k)=A(i,1);
    Y_lidar(k)=A(i,2);
    timestamp_lidar(k)=A(i,3);
    gt_x_lidar(k)=A(i,4);
    gt_y_lidar(k)=A(i,5);
    gt_vx_lidar(k)=A(i,6);
    gt_vy_lidar(k)=A(i,7);
    gt_yaw_lidar(k)=A(i,8);
    gt_yawrate_lidar(k)=A(i,9);
    %radar data
    rho_radar(k)=A(i+1,1);
    pi_radar(k)=A(i+1,2);
    Rdot_radar(k)=A(i+1,3);
    timestamp_radar(k)=A(i+1,4);   
    gt_x_radar(k)=A(i+1,5); 
    gt_y_radar(k)=A(i+1,6);
    gt_vx_radar(k)=A(i+1,7);
    gt_vy_radar(k)=A(i+1,8);
    gt_yaw_radar(k)=A(i+1,9);
    gt_yawrate_radar(k)=A(i+1,10);
end
all_sensor_data=[X_lidar Y_lidar rho_radar pi_radar Rdot_radar];
all_ground_truth=[gt_x_lidar gt_y_lidar gt_vx_lidar gt_vy_lidar gt_yaw_lidar gt_yawrate_lidar gt_x_radar...
    gt_y_radar gt_vx_radar gt_vy_radar gt_yaw_radar gt_yawrate_radar];

%%
syms x y vx vy phi dphi delta_T dts;
state=[x y vx vy phi dphi];
V=sqrt(vx^2+vy^2);
dist=sqrt(x^2+y^2);
Vrel=(x*vx+y*vy)/dist;
% Process model of state vector%
g = [(x+V*(cos(phi))*dts);(y+V*(sin(phi))*dts);vx;vy;phi+(dphi*dts);dphi];  % velocity and yaw rate are constant
%% linearize using jacobian matrix %%
G=jacob(g,x,y,vx,vy,phi,dphi);
dt=0.1;                         %Sampling rate
Q=eye(6);                       %Noise covariance matrix
% measurement matrix
z=[x y dist phi Vrel];
% predict error covariance matrix
P=1000*eye(6);
%Measurement Covariance matrix
R=[0.01 0 0 0 0;0 0.01 0 0 0;0 0 0.01 0 0;0 0 0 1e-6 0;0 0 0 0 0.01]; 
H=jacobian(z,[x y vx vy phi dphi]);    % Jacobian of measurement vector
%% Kalman Filter

%Initial state
x0=[1;1;1;1;1;1];
I=eye(6);
%% State Prediction step
%% Kalman Filter
%Initial state
x0=[1;1;1;1;1;1];
x=x0(1);
y=x0(2);
vx=x0(3);
vy=x0(4);
phi=x0(5);
dphi=x0(6);
I=eye(6);
%% State Prediction step
for i=1:250
x0(1)=dt*(sqrt(x0(3)^2+x0(4)^2)*cos(x0(5)))+x0(1);
x0(2)=dt*(sqrt(x0(3)^2+x0(4)^2)*sin(x0(5)))+x0(2);
x0(3)=x0(3);
x0(4)=x0(4);
x0(5)=dt*x0(6)+x0(5);
x0(6)=x0(6);
%% Jacobian 
a13=dt*x0(3)*cos(x0(5))/sqrt((x0(3)^2+x0(4)^2));
a14=dt*x0(4)*cos(x0(5))/sqrt((x0(3)^2+x0(4)^2));
a23=dt*x0(3)*sin(x0(5))/sqrt((x0(3)^2+x0(4)^2));
a24=dt*x0(4)*sin(x0(5))/sqrt((x0(3)^2+x0(4)^2));
a15=-dt*(sqrt(x0(3)^2+x0(4)^2))*sin(x0(5));
a25=-dt*(sqrt(x0(3)^2+x0(4)^2))*cos(x0(5));

JA=eye(6);
JA(1,3)=a13;
JA(1,4)=a14;
JA(2,3)=a23;
JA(2,4)=a24;
JA(1,5)=a15;
JA(2,5)=a25;
JA(5,6)=dt;

P=JA*P*JA'+Q;

%% Measurement update
hx=[x0(1);x0(2);sqrt(x0(1)^2+x0(2)^2);x0(5);((x0(3)*x0(1))+(x0(4)*x0(2)))/sqrt(x0(1)^2+x0(2)^2)];
    JH=zeros(5,6);
if rem(i,2)==1
    JH(1,1)=1;
    JH(2,2)=1;
else
    h31=x0(1)/sqrt(x0(1)^2+x0(2)^2);
    h32=x0(2)/sqrt(x0(1)^2+x0(2)^2);
    h51=(x0(3)/sqrt(x0(1)^2+x0(2)^2))-((x0(3)*x0(1))+(x0(4)*x0(2)))/sqrt(x0(1)^2+x0(2)^2);
    h52=(x0(4)/sqrt(x0(1)^2+x0(2)^2))-((x0(3)*x0(1))+(x0(4)*x0(2)))/sqrt(x0(1)^2+x0(2)^2);
    h53=x0(1)/sqrt(x0(1)^2+x0(2)^2);
    h54=x0(2)/sqrt(x0(1)^2+x0(2)^2);
    
    JH(3,1)=h31;
    JH(3,2)=h32;
    JH(5,1)=h51;
    JH(5,2)=h52;
    JH(5,3)=h53;
    JH(5,4)=h54;
    JH(4,5)=1;
end

S=JH*P*JH'+R;
K=P*JH'*inv(S);             %Kalman Gain
Z=all_sensor_data(i,:)';
y=Z-hx;
x0=x0+(K*y);
%Update error covariance
P = (I - (K*JH))*P;
all_estimation(i,:)=x0;
if rem(i,2)==1
    %LIDAR OUTPUTS
fprintf('LIDAR OUTPUTS: ')
fprintf('timestamp: ')
disp(A(i,3));
fprintf('LIDAR x and y: ')
disp([A(i,1) A(i,2)]);
fprintf('Prediction: ')
disp(x0');
fprintf('Ground Truth: ')
disp(all_ground_truth(i,[1:6]));
else
    %RADAR OUTPUTS
fprintf('RADAR OUTPUTS: ')
fprintf('timestamp: ')
disp(A(i,4));
fprintf('RADAR distance, heading and relative velocity: ')
disp([A(i,1) A(i,2) A(i,3)]);
fprintf('Prediction: ')
disp(x0');
fprintf('Ground Truth: ')
disp(all_ground_truth(i,[7:12]));  
end
end
%%          PLOTS
figure;
plot(all_estimation(:,1),all_estimation(:,2),'Linewidth',1.5);
hold on;
plot(all_ground_truth(:,1),all_ground_truth(:,2),'Linewidth',1.5);
title('Plot for estimated value vs ground truth value');
legend('Estimated Value','Ground Truth Value');
%% Jacobian Function
function G=jacob(g,x,y,vx,vy,phi,dphi)
%g = [(x+V*(cos(phi))*dts);(y+V*(sin(phi))*dts);vx;vy;phi+(dphi*dts);dphi];  % velocity and yaw rate are constant
%% linearize using jacobian matrix %%
G=jacobian(g,[x y vx vy phi dphi]);
end





