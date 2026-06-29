%% Classification will be done with 3 classifiers:
% 1. KNN
% 2. Naive Bayes
% 3. SVM
%% Import Data
clear all

data = readtable("clean_data.csv",'VariableNamingRule','preserve');

%% K-Nearest Neighbor
classification_model = fitcknn(data,'num','Distance','euclidean');
classifier_KNN = 'KNN';
% I decided to use all the data to train/test this model, for now. I'll see
% how it holds up and if it overfits later

% update: I decided that because the 

cv = cvpartition(classification_model.NumObservations, 'KFold', 5);
cross_validated_model_KNN = crossval(classification_model, 'CVPartition', cv);
Predictions = kfoldPredict(cross_validated_model_KNN);

Results_KNN = confusionmat(cross_validated_model_KNN.Y, Predictions);
% I'm using a pre-written function made by someone else on the matlab
% forums to do this, since it's less time consuming than writing my own
Evaluation_results_KNN = confusion_matrix_stats(cross_validated_model_KNN.Y,Predictions);

%% Naive Bayes

classification_model = fitcnb(data,'num');
classifier_NB = 'NB';

cv = cvpartition(classification_model.NumObservations, 'KFold', 5);
cross_validated_model_NB = crossval(classification_model, 'CVPartition', cv);
Predictions = kfoldPredict(cross_validated_model_NB);

Results_NB = confusionmat(cross_validated_model_NB.Y, Predictions);
Evaluation_results_NB = confusion_matrix_stats(cross_validated_model_NB.Y,Predictions);

%% SVM

classification_model = fitcsvm(data,'num');
classifier_SVM = 'SVM';

cv = cvpartition(classification_model.NumObservations, 'KFold', 5);
cross_validated_model_SVM = crossval(classification_model, 'CVPartition', cv);
Predictions = kfoldPredict(cross_validated_model_SVM);

Results_SVM = confusionmat(cross_validated_model_SVM.Y, Predictions);
Evaluation_results_SVM = confusion_matrix_stats(cross_validated_model_SVM.Y,Predictions);

%% Documenting Accuracy Evaluation Results

eval_results = {Evaluation_results_KNN,Evaluation_results_NB,Evaluation_results_SVM};
classifier_ids = {classifier_KNN,classifier_NB,classifier_SVM};

fileID = fopen('Project 1 - UCI Heart Disease\comparison_classification_results.csv','a');

for i=1:length(eval_results)
    
    eval_struct = eval_results{i};
    classifier = classifier_ids{i};

    fprintf(fileID,'%s %f \n',classifier,mean(eval_struct.accuracy(1)));

    switch i
        case 1
            fprintf(fileID,'%s \n','ConfusionMat_KNN');
        case 2
            fprintf(fileID,'%s \n','ConfusionMat_NB');
        case 3
            fprintf(fileID,'%s \n','ConfusionMat_SVM');
    end
    
    fprintf(fileID,'%d %d \n %d %d \n', eval_struct.confusionMat(1,1),eval_struct.confusionMat(1,2),eval_struct.confusionMat(2,1),eval_struct.confusionMat(2,2));

end

fclose(fileID);

%% ROC Curves

cv_models = {cross_validated_model_KNN,cross_validated_model_NB,cross_validated_model_SVM};

for i=1:length(cv_models)
    model = cv_models{i};

    [~, scores] = kfoldPredict(model);
    [X,Y,~,area_under_curve] = perfcurve(model.Y,scores(:,2),1);
    % scores(:,2) is the confidence score for positive scored class,
    % 1 indicates it's the positive one

    figure
    plot(X,Y)
    xlabel('False positive rate') 
    ylabel('True positive rate')

    AUC_str = num2str(area_under_curve);

    switch i
        case 1
            title(append("ROC of KNN - AUC ="," ",AUC_str))
            file_name = 'ROC_KNN.png';
        case 2
            title(append("ROC of NB - AUC ="," ",AUC_str))
            file_name = 'ROC_NB.png';
        case 3
            title(append("ROC of SVM - AUC ="," ",AUC_str))
            file_name = 'ROC_SVM.png';
    end

    name_img = append('Project 1 - UCI Heart Disease\',file_name);

    saveas(gcf,name_img)
end