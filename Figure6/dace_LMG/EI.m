function [ExpImp] = EI(y_min,y_hat,S2_hat)

% Expected improvement
% Mingyu Lee, KAIST, IDOL LAB
% Last modified: 20211019
% Forrester, A., Sobester, A., & Keane, A. (2008). Engineering design via surrogate modelling: a practical guide. John Wiley & Sons.
% S2_hat: MSE

S2_hat = max(0,S2_hat);
S_hat = sqrt(S2_hat);

if S2_hat == 0
    ExpImp = 0;
else
    EI_first_term = (y_min-y_hat)*(0.5+0.5*erf((1/sqrt(2))*...
        ((y_min-y_hat)/S_hat)));
    EI_second_term = S_hat*(1/sqrt(2*pi))*exp(-(1/2)*...
        ((y_min-y_hat)^2/S2_hat));
    ExpImp = EI_first_term + EI_second_term;
end

% S_hat = sqrt(S2_hat);
% if S_hat == 0
%     ExpImp = 0;
% else
%     EI_first_term = (y_min-y_hat)*normcdf(((y_min-y_hat)/S_hat),0,1);
%     EI_second_term = S_hat*normpdf(((y_min-y_hat)/S_hat),0,1);
%     ExpImp = EI_first_term + EI_second_term;
% end

end

