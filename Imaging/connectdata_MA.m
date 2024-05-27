% Merge single-angle MAT files into multi-angle.
na = UserSet.angle_num;
f_0 = strcat(path_rf,RF_name,'_1.mat');
tp = load(f_0);
if mode == 1
    CombinedData = tp.Label;
    for i=2:na
        tt = strcat(path_rf,RF_name,'_',num2str(i),'.mat');
        f_t = load(tt);
        f_t = f_t.Label;
        CombinedData = cat(3,CombinedData,f_t);
    end
elseif mode == 2
    CombinedData = tp.Data;
    for i=2:na
        tt = strcat(path_rf,RF_name,'_',num2str(i),'.mat');
        f_t = load(tt);
        f_t = f_t.Data;
        CombinedData = cat(3,CombinedData,f_t);
    end
else
    CombinedData = tp.predict;
    for i=2:na
        tt = strcat(path_rf,RF_name,'_',num2str(i),'.mat');
        f_t = load(tt);
        f_t = f_t.predict;
        CombinedData = cat(3,CombinedData,f_t);
    end
end
DataFA = CombinedData;

