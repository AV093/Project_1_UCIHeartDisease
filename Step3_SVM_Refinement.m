%% Import Data
clear all

data = readtable("clean_data.csv",'VariableNamingRule','preserve');

% After thorough research the least important heart disease predictors are:
% trestbps, Female (sex), age, chol
% will make new variable without those & see what it does:
data.trestbps = [];
data.Female = [];
data.age = [];
data.chol = [];
%% Refinement by Optimizing Hyperparameters

%ORIGINAL RESULTS --> No columns removed
% 0.85676 accuracy
% Best estimated feasible point (according to models):
% BoxConstraint    KernelScale    KernelFunction    PolynomialOrder    Standardize
% _____________    ___________    ______________    _______________    ___________
% 
% 0.016973           NaN          polynomial             2              false   
% 
% Estimated objective function value = 0.11984
% Estimated function evaluation time = 0.067327

%MODIFIED RESULTS
% 0.8728 accuracy
% Best estimated feasible point (according to models):
% BoxConstraint    KernelScale    KernelFunction    PolynomialOrder    Standardize
% _____________    ___________    ______________    _______________    ___________
% 
% 0.0010013          NaN          polynomial             4              false   
% 
% Estimated objective function value = 0.13491
% Estimated function evaluation time = 0.087161

classification_model = fitcsvm(data,'num','BoxConstraint',0.0010013,'KernelFunction','polynomial','PolynomialOrder',4,'Standardize',false);
classifier_SVM = 'SVM';

cv = cvpartition(classification_model.NumObservations, 'KFold', 5);
cross_validated_model = crossval(classification_model, 'CVPartition', cv);
Predictions = kfoldPredict(cross_validated_model);

Results_SVM = confusionmat(cross_validated_model.Y, Predictions);
Evaluation_results_SVM = confusion_matrix_stats(cross_validated_model.Y,Predictions);

%% Confusion Matrix Chart
figure
confusionchart(cross_validated_model.Y, Predictions);
eval_acc_str = num2str(Evaluation_results_SVM.accuracy(1));
title(append("SVM Confusion Matrix - Accuracy ="," ",eval_acc_str));

saveas(gcf,'Project 1 - UCI Heart Disease\confusionMat_SVM.png')

%% ROC SVM Chart

[~, scores] = kfoldPredict(cross_validated_model);
[X,Y,~,area_under_curve] = perfcurve(cross_validated_model.Y,scores(:,2),1);

figure
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')

AUC_str = num2str(area_under_curve);

title(append("ROC of SVM - Modified Kernel - AUC ="," ",AUC_str));

saveas(gcf,'Project 1 - UCI Heart Disease\ROC_SVM_MOD.png')
