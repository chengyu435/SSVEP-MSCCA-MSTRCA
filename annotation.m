%20240428 程宇
%chatgpt解读代码并注释


clear all;                    % 清空工作区
close all;                   % 关闭所有图形窗口
load('model.mat')            % 加载模型数据

tic                          % 开始计时

% Chebyshev Type I filter design
for k=1:num_of_subbands      % 遍历每个子带
    Wp = [(8*k)/(Fs/2) 90/(Fs/2)];       % 通带截止频率
    Ws = [(8*k-2)/(Fs/2) 100/(Fs/2)];    % 阻带截止频率
    [N,Wn] = cheb1ord(Wp,Ws,3,40);       % 计算滤波器的阶数和截止频率
    [subband_signal(k).bpB,subband_signal(k).bpA] = cheby1(N,0.5,Wn);  % 设计 Chebyshev Type I 滤波器
end

% Notch filter design
Fo = 50;                      % 基频
Q = 35;                       % Q 值
BW = (Fo/(Fs/2))/Q;           % 带宽

[notchB,notchA] = iircomb(Fs/Fo,BW,'notch');  % 设计 notch 滤波器
seed = RandStream('mt19937ar','Seed','shuffle');  % 初始化随机数生成器

sig_len=TW_p(tw_length);      % 计算信号长度

clear y_sb                   % 清除变量 y_sb
tic                          % 开始计时

% Load single trial data for testing
load('S1.mat');

target=1;                    % 选择目标类别
eeg=data(ch_used,floor(0.5*Fs)+1:floor(0.5*Fs+latencyDelay)+sig_len,target,1);  % 提取单个试验数据

[d1_,d2_,d3_,d4_]=size(eeg);  % 提取数据的维度信息
d1=d3_;d2=d4_;d3=d1_;d4=d2_;   % 更新变量名
no_of_class=d1;                % 类别数
n_ch = d3;                     % 通道数

% Preprocess the test data
for i=1:1:d1                  % 遍历每个类别
    for j=1:1:d2              % 遍历每个试验
        % Frequency band preprocessing
        y0=reshape(eeg(:,:,i,j),d3,d4);  % 重塑数据形状
        y = filtfilt(notchB, notchA, y0.'); %应用notch滤波器
        y = y.';                % 将数据转置为原始形状
        for sub_band=1:num_of_subbands   % 遍历每个子带
            % FB preprocessing
            for ch_no=1:d3      % 遍历每个通道
                tmp2=filtfilt(subband_signal(sub_band).bpB,subband_signal(sub_band).bpA,y(ch_no,:)); %应用子带滤波器
                y_sb(ch_no,:) = tmp2(latencyDelay+1:latencyDelay+sig_len); %截取滤波后的信号长度
            end
            test_signal=reshape(y_sb,d3,length(y_sb),1,1);  %重塑信号形状
        end

    end
end

% Preprocess the signal data
load('S1.mat');              % 重新加载数据

eeg=data(ch_used,floor(0.5*Fs)+1:floor(0.5*Fs+latencyDelay)+sig_len,:,:);  %提取信号数据

[d1_,d2_,d3_,d4_]=size(eeg);  % 提取数据维度信息
d1=d3_;d2=d4_;d3=d1_;d4=d2_;   % 更新变量名
no_of_class=d1;                % 类别数
n_ch = d3;                     % 通道数

if sn==1   %如果是第一次试验
    for sub_band=1:num_of_subbands   %遍历每个子带
        subband_signal(sub_band).SSVEPdata = zeros(n_ch,sig_len,d2,d1);  %初始化SSVEPdata数组
        subband_signal(sub_band).signal_template = zeros(n_ch,sig_len,d1);  %初始化信号模板数组
    end
end

% Preprocessing of signal data
for i=1:1:d1                %遍历每个类别
    for j=1:1:d2            %遍历每个试验
        % Frequency band preprocessing
        y0=reshape(eeg(:,:,i,j),d3,d4);   %重塑数据形状
        y = filtfilt(notchB, notchA, y0.');  %应用notch滤波器
        y = y.';              %将数据转置为原始形状
        for sub_band=1:num_of_subbands   %遍历每个子带
            % FB preprocessing
            for ch_no=1:d3    %遍历每个通道
                tmp2=filtfilt(subband_signal(sub_band).bpB,subband_signal(sub_band).bpA,y(ch_no,:));  %应用子带滤波器
                y_sb(ch_no,:) = tmp2(latencyDelay+1:latencyDelay+sig_len);  %截取滤波后的信号长度
            end
            subband_signal(sub_band).SSVEPdata(:,:,j,i)=reshape(y_sb,d3,length(y_sb),1,1);  %重塑信号形状并存储到SSVEPdata数组中
        end
        
    end
end

% Initialization
for sub_band=1:num_of_subbands   %遍历每个子带
    subband_signal(sub_band).SSVEPdata=subband_signal(sub_band).SSVEPdata(:,:,:,target_order); %按照8.0、8.2、8.4等顺序对数据进行排序
end

FB_coef=FB_coef0'*ones(1,n_sti);  %初始化FB_coef

seq_0=zeros(d2,num_of_trials);     %初始化seq_0数组

idx_traindata=seq1;                  %设置训练数据的索引
idx_testdata=1:n_run;                %设置测试数据的索引
idx_testdata(seq1)=[];

% Classifier, predict template for each target trial
for i=1:no_of_class                  %遍历每个类别
    for k=1:num_of_subbands          %遍历每个滤波器子带
        if length(idx_traindata)>1   %如果训练数据的长度大于1
            subband_signal(k).signal_template(:,:,i)=mean(subband_signal(k).SSVEPdata(:,:,:,i),3); %计算模板的平均值
        else
            subband_signal(k).signal_template(:,:,i)=subband_signal(k).SSVEPdata(:,:,:,i);  %直接使用模板数据
        end
    end
end

% Classification
for sub_band=1:num_of_subbands       %遍历每个子带
    if (is_center_std==1)            %如果使用中心化和标准化
        test_signal=test_signal-mean(test_signal,2)*ones(1,length(test_signal));  %中心化
        test_signal=test_signal./(std(test_signal')'*ones(1,length(test_signal))); %标准化
    end
    for j=1:no_of_class              %遍历模板
        template=subband_signal(sub_band).signal_template(:,[1:sig_len],j);  %获取模板
        if (is_center_std==1)        %如果使用中心化和标准化
            template=template-mean(template,2)*ones(1,length(template));    %中心化
            template=template./(std(template')'*ones(1,length(template)));  %标准化
        end

        % Generate the sine-cosine reference signal
        ref1=ref_signal_nh(sti_f(j),Fs,pha_val(j),sig_len,num_of_harmonics); %生成正弦余弦参考信号

        % =============== ms-eCCA ===============
        if (enable_bit(2)==1)         %如果启用ms-eCCA
            if (i==1)                 %如果是第一个类别
                % find the indices of neighboring templates
                d0=floor(num_of_signal_templates/2); %计算相邻模板的索引范围
                %%=================寻找最近的12个模板=============
                if j<=d0
                    template_st=1;
                    template_ed=num_of_signal_templates;
                elseif ((j>d0) && j<(d1-d0+1))
                    template_st=j-d0;
                    template_ed=j+(num_of_signal_templates-d0-1);
                else
                    template_st=(d1-num_of_signal_templates+1);
                    template_ed=d1;
                end

                mscca_template=[];
                mscca_ref=[];
                template_seq=[template_st:template_ed];

                % Concatenation of the templates (or sine-cosine references)
                %%================先遍历，然后存储mscca_template和mscca_ref，总共是[6480,6]的数据=============
                for n_temp=1:num_of_signal_templates%遍历12个模板[1:12]
                    template0=subband_signal(sub_band).signal_template(:,1:sig_len,template_seq(n_temp));
                    if (is_center_std==1)
                        template0=template0-mean(template0,2)*ones(1,length(template0));
                        template0=template0./(std(template0')'*ones(1,length(template0)));
                    end
                    ref0=ref_signal_nh(sti_f(template_seq(n_temp)),Fs,pha_val(template_seq(n_temp)),sig_len,num_of_harmonics);
                    mscca_template=[mscca_template;template0'];
                    mscca_ref=[mscca_ref;ref0'];
                end
                % ========mscca spatial filter=====
                [Wx1,Wy1,cr1]=canoncorr(mscca_template,mscca_ref(:,1:end)); %计算空间滤波器
                spatial_filter1(sub_band,j).wx1=Wx1(:,1)';  %存储空间滤波器
                spatial_filter1(sub_band,j).wy1=Wy1(:,1)';

            end


            cr1=corrcoef((spatial_filter1(sub_band,j).wx1*test_signal)',(spatial_filter1(sub_band,j).wy1*ref1)'); %计算相关系数
            cr2=corrcoef((spatial_filter1(sub_band,j).wx1*test_signal)',(spatial_filter1(sub_band,j).wx1*template)'); %计算相关系数
            %遍历j=[1:40]的模板，每个模板生成一个trial
            msccaR(sub_band,j)=sign(cr1(1,2))*cr1(1,2)^2+sign(cr2(1,2))*cr2(1,2)^2;  %计算相似度分数
        else
            msccaR(sub_band,j)=0;
        end

    end

end%遍历每一个test数据的trial，计算相似性

msccaR1=sum((msccaR).*FB_coef,1);  %计算加权相似度
[~,idx]=max(msccaR1);            %获取最大相似度对应的类别索引
idx                              %输出类别索引

toc                              %结束计时
