function  [para, fea_w] = chooseFeatureSelectAlgorithm(X_tr,Y_tr,opt)

currentPath = pwd;          
%% start classification
switch opt.algorithm
    case 'optimize_W_b_a'
        addpath([currentPath,'\model\QMVFS']);
        para.tol = 1e-5;
        para.eta = 1e-6;
        [alpha, beta, fea_w, obj_values] = optimize_W_b_a(X_tr, Y_tr, opt.lambda, opt.p, opt.gamma, 300, para.eta, para.tol);
        para.alpha = alpha;
        para.beta = beta;
        para.obj_values = obj_values; 
        rmpath([currentPath,'\model\QMVFS']);   
    case 'EQI-BGWO'
        addpath([currentPath,'\model\EQI-BGWO']);
        [para.fitness, fea_w] = main(X_tr, Y_tr);
        rmpath([currentPath,'\model\EQI-BGWO']);
    case 'QSIFS'
        addpath([currentPath,'\model\QSIFS']);
        [fea_w] = main(X_tr, Y_tr);
        para = 'none';
        rmpath([currentPath,'\model\QSIFS']);
    case 'UMFS'
        addpath([currentPath,'\model\CE-UMFS']);
        param.gamma1 = 10;
        param.beta = 10;
        param.lambda1 =0.01; 
        param.lambda2 = 0.1; 
        param.beta2 = 1; 
        [fea_w] = main(X_tr, param.gamma1, param.beta, param.lambda1, param.lambda2, param.beta2);
        para = 'none';
        rmpath([currentPath,'\model\CE-UMFS']);
    case 'GCDUFS'
        addpath([currentPath,'\model\GCDUFS']);
        para.alpha=1e2;%[1e-2,1e-1,1,1e1,1e2];
        para.beta=1e-2;%[1e-2,1e-1,1,1e1,1e2];
        para.gamma=1;%[1e-2,1e-1,1,1e1,1e2];
        para.lammbda=1e-1;
        fea_w = main(X_tr, Y_tr, para);
        rmpath([currentPath,'\model\GCDUFS']);
    
end

