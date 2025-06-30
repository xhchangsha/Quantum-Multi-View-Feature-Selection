clearvars;
opt.algorithm='optimize_W_b_a';%optimize_W_b_a EQI-BGWO QSIFS UMFS GCDUFS
dataname = 'Prokaryotic';
load(fullfile('dataset', [dataname, '.mat']));
V = length(X);
XX = [];
for v = 1:V
    XX = [XX, X{v}];
end
data=[XX, Y];     
[~, xxd] = size(XX);

%load kernel matrix obtained through the circuit
% k represents the number of kernel circuits corresponding to each view.
ken = '3';
if strcmp(opt.algorithm , 'optimize_W_b_a')
    k_dataname = ['k_', dataname,'_',ken];
    load(fullfile('dataset', [k_dataname, '.mat']));
    [kn, kd] = size(k_X);
end

SelectFeaNum=zeros(xxd,1);

all_fea_w = cell(1,10);
all_para = cell(1,10);
all_matrics = cell(1,10);
all_acc = zeros(1, 10);
all_macro_precision = zeros(1, 10);
all_macro_recall = zeros(1, 10);
all_macro_f1 = zeros(1, 10);

no_select_num = 0;%address the situation where no features are selected for a certain fold in EQI-BGWO of the baseline approach

all_indices=crossvalind('Kfold',size(data,1),10);
for k=1:10
    opt.k = k;
    testnum=(all_indices==k);
    trainnum=~testnum;
    if strcmp(opt.algorithm , 'optimize_W_b_a')||strcmp(opt.algorithm , 'UMFS')||strcmp(opt.algorithm , 'GCDUFS')
        for v =1:V
            X_test{v}=X{v}(testnum==1,:);
            X_train{v}=X{v}(trainnum==1,:); 
        end
        if strcmp(opt.algorithm , 'optimize_W_b_a')
            for i=1:kn
                for j = 1:kd
                    K_train{i, j}=k_X{i,j}(trainnum==1,trainnum==1);
                end
            end
        end
    end     
    XX_test=XX(testnum==1,:);
    XX_train=XX(trainnum==1,:);  
    
    Y_test=Y(testnum==1,:);
    Y_train=Y(trainnum==1,:);
    
    if strcmp(opt.algorithm , 'optimize_W_b_a')
        opt.lambda = 300;
        opt.p = 10;
        opt.gamma = 3;
        [para, fea_w]=chooseFeatureSelectAlgorithm(X_train,K_train,opt);
        %SelectFeaIdx = find(fea_w > 1e-6);
        para.percent = 0.1;
        para.lambda = opt.lambda;
        para.p = opt.p;
        para.gamma = opt.gamma;
        SelectFeaIdx =[];
        for v = 1:V
            [~, xtrd] = size(X_train{v});
            normW = sqrt(sum(fea_w{v}.*fea_w{v},2));
            [T_Weight, T_sorted_features] = sort(normW,'descend');
            Num_SelectFeaLY = floor(para.percent*xtrd);
            dis = 0;
            if v>1
                [~, dis] = size(X_train{v-1});
            end
            SelectFeaIdx = [SelectFeaIdx;T_sorted_features(1:Num_SelectFeaLY)+dis];
        end       
    elseif strcmp(opt.algorithm , 'UMFS')||strcmp(opt.algorithm , 'GCDUFS')
        [para, fea_w]=chooseFeatureSelectAlgorithm(X_train,Y_train,opt);
        [T_Weight, T_sorted_features] = sort(fea_w,'descend');
        percent = 0.8;
        Num_SelectFeaLY = floor(percent*xxd);
        SelectFeaIdx = T_sorted_features(1:Num_SelectFeaLY);
    else
        [para, fea_w]=chooseFeatureSelectAlgorithm(XX_train,Y_train,opt);
        SelectFeaIdx = find(fea_w == 1);
    end

    all_fea_w{1,k} = fea_w;
    all_para{1,k} = para;
    if ~isempty(SelectFeaIdx)  
        SelectFeaNum(SelectFeaIdx)=SelectFeaNum(SelectFeaIdx)+1;
        X_trainwF = XX_train(:,SelectFeaIdx);
        X_testwF = XX_test(:,SelectFeaIdx); 

        model = fitcecoc(X_trainwF, Y_train);
        predictedLabels = predict(model, X_testwF);
        metrics = EvaluationMetrics(predictedLabels, Y_test);
        all_matrics{1, k} = metrics;
        all_acc(k) = metrics.accuracy;
        all_macro_precision(k) = metrics.macro_precision;
        all_macro_recall(k) = metrics.macro_recall;
        all_macro_f1(k) = metrics.macro_f1;
    else
        no_select_num = no_select_num+1;
    end    
end

total_acc = sum(all_acc(:))/(10-no_select_num);
total_macro_precision = sum(all_macro_precision(:))/(10-no_select_num);
total_macro_recall = sum(all_macro_recall(:))/(10-no_select_num);
total_macro_f1 = sum(all_macro_f1(:))/(10-no_select_num);

[order_select_num,order_select_id] = sort(SelectFeaNum,'descend');
save(['result\',char(dataname),'_svm_',char(opt.algorithm),'_best_result_',num2str(total_acc),'_',num2str(total_macro_precision),'_',num2str(total_macro_recall),'_',num2str(total_macro_f1),'.mat'],'all_indices', 'all_fea_w', 'all_para', 'all_matrics', 'all_acc', 'all_macro_precision', 'all_macro_recall', 'all_macro_f1','order_select_num','order_select_id');



