function metrics = EvaluationMetrics(predicted_labels, true_labels)
    % 计算多分类和二分类任务的评价指标
    % 
    % 参数:
    %   predicted_labels - 预测类别向量 (N x 1)
    %   true_labels - 真实类别向量 (N x 1)
    %
    % 返回:
    %   metrics - 结构体，包含各种评估指标

classes = unique([true_labels; predicted_labels]);  % 统一类别
num_classes = length(classes);
C = confusionmat(true_labels, predicted_labels, 'Order', classes);


    % % 获取类别数
    % classes = unique(true_labels);
    % num_classes = length(classes);
    % 
    % % 计算混淆矩阵
    % C = confusionmat(true_labels, predicted_labels);

    % 初始化变量
    TP = zeros(num_classes, 1);  % True Positives
    FP = zeros(num_classes, 1);  % False Positives
    FN = zeros(num_classes, 1);  % False Negatives
    TN = zeros(num_classes, 1);  % True Negatives

    % 计算 TP, FP, FN, TN
    for i = 1:num_classes
        TP(i) = C(i, i);
        FP(i) = sum(C(:, i)) - TP(i);
        FN(i) = sum(C(i, :)) - TP(i);
        TN(i) = sum(C(:)) - (TP(i) + FP(i) + FN(i));
    end

    % 避免 0 除错误
    safe_divide = @(a, b) a ./ (b + (b == 0));

    % 计算 Precision, Recall (Sensitivity), F1 Score
    precision = safe_divide(TP, TP + FP);
    recall = safe_divide(TP, TP + FN);  % Recall = Sensitivity
    f1_score = safe_divide(2 * (precision .* recall), precision + recall);

    % 计算宏 (Macro) 指标
    macro_precision = mean(precision);
    macro_recall = mean(recall);
    macro_f1 = mean(f1_score);

    % 计算微 (Micro) 指标
    micro_precision = safe_divide(sum(TP), sum(TP + FP));
    micro_recall = safe_divide(sum(TP), sum(TP + FN));
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall);

    % 计算加权 (Weighted) 指标

    class_counts = sum(C, 2);  % 每个类别的样本数
    weights = class_counts / sum(class_counts);
    % weights = weights(:);  % 强制转换为列向量
    weighted_precision = sum(weights .* precision);
    weighted_recall = sum(weights .* recall);
    weighted_f1 = sum(weights .* f1_score);

    % 计算准确率 (Accuracy)
    accuracy = sum(TP) / sum(C(:));

    % 计算二分类任务的 Specificity, Sensitivity, AUC
    if num_classes == 2
        specificity = safe_divide(TN, TN + FP);  % 特异度 (Specificity)
        sensitivity = recall;  % 召回率即 Sensitivity
        fpr = safe_divide(FP, FP + TN);  % 假阳性率 (FPR)
        auc = (sensitivity(1) + specificity(1)) / 2;  % AUC 计算

        % 记录二分类指标
        metrics.sensitivity = sensitivity(1);  % 选择第一个类别的 Sensitivity
        metrics.specificity = specificity(1);
        metrics.fpr = fpr(1);
        metrics.auc = auc;
    else
        metrics.sensitivity = NaN;
        metrics.specificity = NaN;
        metrics.fpr = NaN;
        metrics.auc = NaN;
    end

    % 记录所有指标
    metrics.accuracy = accuracy;
    metrics.macro_precision = macro_precision;
    metrics.macro_recall = macro_recall;
    metrics.macro_f1 = macro_f1;
    metrics.micro_precision = micro_precision;
    metrics.micro_recall = micro_recall;
    metrics.micro_f1 = micro_f1;
    metrics.weighted_precision = weighted_precision;
    metrics.weighted_recall = weighted_recall;
    metrics.weighted_f1 = weighted_f1;

    % 打印结果
    fprintf('Accuracy: %.4f\n', accuracy);
    fprintf('Macro Precision: %.4f\n', macro_precision);
    fprintf('Macro Recall: %.4f\n', macro_recall);
    fprintf('Macro F1 Score: %.4f\n', macro_f1);
    fprintf('Micro Precision: %.4f\n', micro_precision);
    fprintf('Micro Recall: %.4f\n', micro_recall);
    fprintf('Micro F1 Score: %.4f\n', micro_f1);
    fprintf('Weighted Precision: %.4f\n', weighted_precision);
    fprintf('Weighted Recall: %.4f\n', weighted_recall);
    fprintf('Weighted F1 Score: %.4f\n', weighted_f1);

    if num_classes == 2
        fprintf('Sensitivity (Recall): %.4f\n', metrics.sensitivity);
        fprintf('Specificity: %.4f\n', metrics.specificity);
        fprintf('False Positive Rate (FPR): %.4f\n', metrics.fpr);
        fprintf('AUC: %.4f\n', metrics.auc);
    end
end
