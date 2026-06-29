clear all

%% This project focuses on multivariate binary classification of the UCI Heart Disease dataset

data = readtable('heart_disease_uci.csv');
%% Data Preprocessing

%% Column Removal

% getting rid of ID & dataset, which are irrelevant
data = data(:,2:end);
data.dataset = [];

% as a general rule of thumb, columns with 50% or more missing are removed,
% the rest are imputed
% slope and ca will be removed
data.ca = [];
data.thal = [];

%% Handling missing values
miss_cols = sum(ismissing(data));

% class based value filling
class_0 = data.num == 0;
class_1 = data.num > 0;
% any value above 1 is essentially meaningless since this dataset is
% mainly used for binary classification rather than multiclassification
% problems
data.num(class_1) = 1;

%% Non-categorical

% trestbps
class_0_mean_trestbps = mean(data.trestbps(class_0), 'omitnan');
class_1_mean_trestbps = mean(data.trestbps(class_1), 'omitnan');

data.trestbps(class_0) = fillmissing(data.trestbps(class_0), 'constant', class_0_mean_trestbps);
data.trestbps(class_1) = fillmissing(data.trestbps(class_1), 'constant', class_1_mean_trestbps);

% chol % change to median if there are outliers, mean is prone to
% overrepresenting outlier values
class_0_median_chol = median(data.chol(class_0), 'omitnan');
class_1_median_chol = median(data.chol(class_1), 'omitnan');

data.chol(class_0) = fillmissing(data.chol(class_0), 'constant', class_0_median_chol);
data.chol(class_1) = fillmissing(data.chol(class_1), 'constant', class_1_median_chol);

% thalch
class_0_mean_thalch = mean(data.thalch(class_0), 'omitnan');
class_1_mean_thalch = mean(data.thalch(class_1), 'omitnan');

data.thalch(class_0) = fillmissing(data.thalch(class_0), 'constant', class_0_mean_thalch);
data.thalch(class_1) = fillmissing(data.thalch(class_1), 'constant', class_1_mean_thalch);

% oldpeak % change to median because there are significant outliers
class_0_median_oldpeak = median(data.oldpeak(class_0), 'omitnan');
class_1_median_oldpeak = median(data.oldpeak(class_1), 'omitnan');

data.oldpeak(class_0) = fillmissing(data.oldpeak(class_0), 'constant', class_0_median_oldpeak);
data.oldpeak(class_1) = fillmissing(data.oldpeak(class_1), 'constant', class_1_median_oldpeak);


%% Categorical Data

% fbs
data.fbs = categorical(data.fbs);

class_0_mode_fbs = mode(data.fbs(class_0)); % mode used because categorical
class_1_mode_fbs = mode(data.fbs(class_1));

data.fbs(class_0) = fillmissing(data.fbs(class_0), 'constant', cellstr(class_0_mode_fbs));
data.fbs(class_1) = fillmissing(data.fbs(class_1), 'constant', cellstr(class_1_mode_fbs));


% exang
data.exang = categorical(data.exang); % converts into a categorical array

class_0_mode_exang = mode(data.exang(class_0));
class_1_mode_exang = mode(data.exang(class_1));

data.exang(class_0) = fillmissing(data.exang(class_0), 'constant', cellstr(class_0_mode_exang));
data.exang(class_1) = fillmissing(data.exang(class_1), 'constant', cellstr(class_1_mode_exang));


% slope
data.slope = categorical(data.slope); % converts into a categorical array

class_0_mode_slope = mode(data.slope(class_0));
class_1_mode_slope = mode(data.slope(class_1));

data.slope(class_0) = fillmissing(data.slope(class_0), 'constant', cellstr(class_0_mode_slope));
data.slope(class_1) = fillmissing(data.slope(class_1), 'constant', cellstr(class_1_mode_slope));

data.slope = cellstr(data.slope); % returns it to cellstr so it can be encoded

%% Removing the missing rows

% restecg
missing_restecg = any(ismissing(data.restecg),2);
data = data(~missing_restecg,:);
% done last because removing rows before is more effort than it's worth cus
% i'll have to rewrite my classes again

% dropping all rows of chol with 0 because it's missing values that
% wouldn't be accurate to the dataset if I filled in 100+ values
data(data.chol == 0, :) = [];
% note: ismissing returns 62 for the column of oldpeak, but 0 is a valid
% value for ST depression, and is seen often in healthy patients

%% Encoding categorical values

% unordered categorical values

% encoding sex
data = unordered_categorical_data_encoding(data,data.sex);
data.sex = [];
% drop male column leave it as a reference point
data.Male = [];

% encoding cp
data = unordered_categorical_data_encoding(data,data.cp);
data.cp = [];
% drop non-anginal column leave it as a reference point
data.("non-anginal") = [];

% encoding restecg
data = unordered_categorical_data_encoding(data,data.restecg);
data.restecg = [];

% drop normal column to leave it as a reference point
data.normal = [];

% ordered categorical values
% encoding slope
new_variable = categorical_data_to_numbers(data.slope,{'downsloping','flat','upsloping'},[-1 0 1]);
data.slope = new_variable;

% binary values of True/False
% fbs
data.fbs = cellstr(data.fbs);
new_variable = categorical_data_to_numbers(data.fbs,{'TRUE','FALSE'},[1 0]);
data.fbs = new_variable;

% exang
data.exang = cellstr(data.exang);
new_variable = categorical_data_to_numbers(data.exang,{'TRUE','FALSE'},[1 0]);
data.exang = new_variable;

%% Removing Outliers
% class based value filling
% remake classes due to previous class_0 and class_1 having indices of
% early iterations
class_0 = data.num == 0;
class_1 = data.num > 0;

% trestbps
figure
boxplot(data.trestbps,data.num);
title('trestbps boxplot for outlier IDing')

trestbps_0 = filloutliers(data.trestbps(class_0),'clip','quartiles');
trestbps_1 = filloutliers(data.trestbps(class_1),'clip','quartiles');

data.trestbps(class_0) = trestbps_0;
data.trestbps(class_1) = trestbps_1;

% chol
figure
boxplot(data.chol,data.num);
title('chol boxplot for outlier IDing')

cholest_0 = filloutliers(data.chol(class_0),'clip','quartiles');
cholest_1 = filloutliers(data.chol(class_1),'clip','quartiles');

data.chol(class_0) = cholest_0;
data.chol(class_1) = cholest_1;

% oldpeak
figure
boxplot(data.oldpeak,data.num);
title('oldpeak boxplot for outlier IDing')

oldpeak_0 = filloutliers(data.oldpeak(class_0),'clip','quartiles');
oldpeak_1 = filloutliers(data.oldpeak(class_1),'clip','quartiles');

data.oldpeak(class_0) = oldpeak_0;
data.oldpeak(class_1) = oldpeak_1;

%% It might do well to discretize age, trestbps, and other things based off of medical
% parameters that define risk of heart disease
% I've decided against it, since the dataset is already very small and it
% overcomplicates things

%% Data Standardization
% age, trestbps, chol, thalch, and oldpeak are all on different scales
% standardize to ensure the ML algo understands that they all have equal
% 'influences' in the dataset

% age
stand_age = (data.age - mean(data.age)) / std(data.age);
data.age = stand_age;

% trestbps
stand_trestbps = (data.trestbps - mean(data.trestbps)) / std(data.trestbps);
data.trestbps = stand_trestbps;

% chol
stand_chol = (data.chol - mean(data.chol)) / std(data.chol);
data.chol = stand_chol;

% thalch
stand_thalch = (data.thalch - mean(data.thalch)) / std(data.thalch);
data.thalch = stand_thalch;

% oldpeak
stand_oldpeak = (data.oldpeak - mean(data.oldpeak)) / std(data.oldpeak);
data.oldpeak = stand_oldpeak;

%% Saving cleaned data
writetable(data,'Project 1 - UCI Heart Disease\clean_data.csv')