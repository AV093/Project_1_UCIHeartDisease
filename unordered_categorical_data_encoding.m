function data = unordered_categorical_data_encoding(data,variable)

uniques = unique(variable);

T = table;

[rowD,colD] = size(data);

dummy_array = zeros(rowD,length(uniques));

for i=1:length(uniques)
    dummy_array(:,i) = double(ismember(variable,uniques{i}));
end

[rows,cols] = size(dummy_array);

for i=1:cols
    T1 = array2table(dummy_array(:,i));
    T1.Properties.VariableNames = uniques(i);
    T = [T T1];

end

data = [T data];

end