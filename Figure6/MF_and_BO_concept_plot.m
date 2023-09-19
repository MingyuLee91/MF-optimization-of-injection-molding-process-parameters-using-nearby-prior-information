close all
clear
clc

%% Add the path
addpath('dace_LMG')

%% Obtain the LF data
X_LF = (0:0.1:1)'; % LF input data
Y_LF = 0.4*((6*X_LF-2).^2.*sin(2*(6*X_LF-2)))+10*(0.4*X_LF.^4+0.1*X_LF.^2+0.2*X_LF+0.2)-10; % LF output data

%% Obtain the HF data
X_HF = [0 0.4 0.6 1]'; % HF input data
Y_HF = (6*X_HF-2).^2.*sin(2*(6*X_HF-2)); % HF output data

%% Build the surrogate models
dm = 1;
Lb = 0;
Ub = 1;

Hybridopts = optimoptions('fmincon','Algorithm','sqp','TolFun',1e-6);
Options_opt = optimoptions('ga','TolFun',1e-6,'MaxGenerations',100*dm,'HybridFcn',{@fmincon,Hybridopts});
Options_opt.PopulationSize = 200;
Options_opt.MaxStallGenerations = 100;

dm = size(X_LF,2);
Theta0 = 1e0*ones(1,dm);
Lb_theta = 1e-6*ones(1,dm);
Ub_theta = 1e2*ones(1,dm);
regpoly = @regpoly0;
corr = @corrgauss;
% corr = @corrspline;


[y_min, ~] = min(Y_HF);
% HF Kriging model (Single-fidelity surrogate)
[dmodel_HF] = dacefit_LMG(X_HF,Y_HF,regpoly,corr,Theta0,Lb_theta,Ub_theta);
HF_krig = @(x) predictor_LMG(x,dmodel_HF);

% Hierarchical Kriging model (Multi-fidelity surrogate)
[dmodel_LF] = dacefit_LMG(X_LF,Y_LF,regpoly,corr,Theta0,Lb_theta,Ub_theta);
LF_krig = @(x) predictor_LMG(x,dmodel_LF);

[dmodel_HK] = dacefit_HK_LMG(X_HF,Y_HF,LF_krig,corr,Theta0,Lb_theta,Ub_theta);
HK_krig = @(x) predictor_HK_LMG(x,dmodel_HK);
HK_krig_mse = @(x) predictor_mse_HK_LMG(x,dmodel_HK);
HK_EI = @(x) EI(y_min,HK_krig(x),HK_krig_mse(x));
HK_Neg_EI = @(x) -EI(y_min,HK_krig(x),HK_krig_mse(x));

[x_new_tmp, HK_EI_max, ~, ~, ~, ~] = ga(HK_Neg_EI, dm, [], [], [], [], Lb, Ub, [], Options_opt);

%% Calculate data for plotting
xx = (0:0.001:1)';

Y_true = (6*xx-2).^2.*sin(2*(6*xx-2));

HF_plot = HF_krig(xx);
HK_plot = HK_krig(xx);
for jj=1:size(xx,1)
    EI_plot(jj,:) = HK_EI(xx(jj,:));
end

%% Perform the post-processing (Prediction)
figure1 = figure;
axes1 = axes('Parent',figure1,'FontWeight','bold','FontSize',12,...
    'FontName','Times New Roman');
box(axes1,'on');

plot(xx,Y_true,'k-','LineWidth',1.8)
hold on

plot(xx,HF_plot,'b-.','LineWidth',1.6)
hold on
plot(xx,HK_plot,'k--','LineWidth',1.6)
hold on

plot(X_HF,Y_HF,'k^','MarkerFaceColor','b','MarkerSize',14,'LineWidth',1.6)
hold on
plot(X_LF,Y_LF,'ks','MarkerFaceColor','r','MarkerSize',16,'LineWidth',1.6)

xlabel('\itx','fontname','Times New Roman')
ylabel('\itf\rm(\itx\rm)','fontname','Times New Roman')

legend('True HF function','HF surrogate model (Kriging)','MF surrogate model (HK)',...
    'HF data','LF data','Location','northwest')
set(gca,'fontsize',18)
set(gca, 'FontName', 'Times New Roman')


%% Perform the post-processing (EI)
figure2 = figure;
axes2 = axes('Parent',figure2,'FontWeight','bold','FontSize',12,...
    'FontName','Times New Roman');
box(axes2,'on');

plot(xx,EI_plot,'k--','LineWidth',1.8)
hold on
plot(x_new_tmp,HK_EI_max,'kp','LineWidth',1.8,'MarkerSize',16)

% grid on
ax = gca;
ax.GridLineStyle = '--';
ax.GridAlpha = 0.25;
ax.XAxis.LineWidth = 1.2;
ax.YAxis.LineWidth = 1.2;

xlabel('\itx','fontname','Times New Roman')
ylabel('\itEI\rm(\itx\rm)','fontname','Times New Roman')

ylim([-0.2 4])

legend('EI function','The next updating point','Location','Northwest')
set(gca,'fontsize',18)
set(gca, 'FontName', 'Times New Roman')
