%20240430程宇
%针对在线数据进行单试次数据分类





clear all;
close all;
% Please download the SSVEP benchmark dataset for this code
% Wang, Y., et al. (2016). A benchmark dataset for SSVEP-based brain-computer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(10), 1746-1752.
% Then indicate where the directory of the dataset is :

addpath('..\mytoolbox\');
Fs=300; % sample rate

dataset_no=1;

% str_dir='..\Tsinghua dataset 2016\';
num_of_wn=4;                        % for TDCA
num_of_k=8;                         % for TDCA
num_of_delay=6;                     % for TDCA
latencyDelay = round(0.14*Fs);      % latency
num_of_subj=1;                     % Number of subjects (35 if you have the benchmark dataset)
% ch_used=[48 54 55 56 57 58 61 62 63]; % Pz, PO5, PO3, POz, PO4, PO6, O1,Oz, O2 (in SSVEP benchmark dataset)
ch_used=[1 7 13 14 15 16]; % Pz, PO3, POz, PO4,Oz, O2 (in SSVEP benchmark dataset)
% ch_used=[1 2 3 4 5 6 7 8 9];
num_of_trials=5;                    % Number of training trials (1<=num_of_trials<=5)
% pha_val=[0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 ...
%     0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5 0 0.5 1 1.5]*pi;
% sti_f=[8.0:1:15.0, 8.2:1:15.2,8.4:1:15.4,8.6:1:15.6,8.8:1:15.8];
pha_val=[0 0.5 1 1.5 0 0.5 1 1.5 0]*pi;
sti_f=[8,9,10,11,12,13,14,15,17];
n_sti=length(sti_f);                     % number of stimulus frequencies
[~,target_order]=sort(sti_f);
sti_f=sti_f(target_order);



num_of_harmonics=3;                 % for all cca-based methods
num_of_signal_templates=5;         % for mscca (1<=num_of_signal_templates<=40)
num_of_signal_templates2=2;         % for ms-etrca (1<=num_of_signal_templates<=40)
num_of_r=4;                         % for ecca
num_of_subbands=5;                  % for filter bank analysis
FB_coef0=[1:num_of_subbands].^(-1.25)+0.25; % for filter bank analysis


% About the above parameter, please check the related paper:
% Chen, X., et al. (2015). Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain?Vcomputer interface. Journal of neural engineering, 12(4), 046008.

% time-window length (min_length:delta_t:max_length)
min_length=0.3;
delta_t=0.1;
max_length=1.5;                     % [min_length:delta_t:max_length] 有0.2秒的延迟

TW = min_length:delta_t:max_length;

% TW = 1.5;
TW_p = round(TW*Fs);
enable_bit=[1 0 0 0 0];             % Select the algorithms: bit 1: eCCA, bit 2: ms-eCCA, bit 3: eTRCA, bit 4: ms-eTRCA, bit 5: TDCA, e.g., enable_bit=[1 1 1 1 1]; -> select all four algorithms
% enable_bit=[1 1 1 1 1];             % Select the algorithms: bit 1: eCCA, bit 2: ms-eCCA, bit 3: eTRCA, bit 4: ms-eTRCA, bit 5: TDCA, e.g., enable_bit=[1 1 1 1 1]; -> select all four algorithms
is_center_std=0;                    % 0: without , 1: with (zero mean, and unity standard deviation)

% Chebyshev Type I filter design
for k=1:num_of_subbands
    Wp = [(8*k)/(Fs/2) 90/(Fs/2)];
    Ws = [(8*k-2)/(Fs/2) 100/(Fs/2)];
    [N,Wn] = cheb1ord(Wp,Ws,3,40);
    [subband_signal(k).bpB,subband_signal(k).bpA] = cheby1(N,0.5,Wn);
end
%notch
Fo = 50;
Q = 35;
BW = (Fo/(Fs/2))/Q;

[notchB,notchA] = iircomb(Fs/Fo,BW,'notch');
seed = RandStream('mt19937ar','Seed','shuffle');

for tw_length=1:length(TW)
    sig_len=TW_p(tw_length);

    clear y_sb
    clear y_sb_test
    sn=1;
    tic
    load('S1.mat');%读取训练数据
    load("wang1_0430.mat")
    target=7;
    test_data=eeg(ch_used,(target-1)*1000+floor(0.5*Fs)+1:(target-1)*1000+floor(0.5*Fs+latencyDelay)+sig_len);


    %  pre-stimulus period: 0.5 sec
    %  latency period: 0.14 sec
    eeg=data(ch_used,floor(0.5*Fs)+1:floor(0.5*Fs+latencyDelay)+sig_len,:,:);

    [d1_,d2_,d3_,d4_]=size(eeg);
    d1=d3_;d2=d4_;d3=d1_;d4=d2_;
    no_of_class=d1;
    n_ch = d3;
    % d1: num of stimuli
    % d2: num of trials
    % d3: num of channels % Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
    % d4: num of sampling points
    if sn==1
        for sub_band=1:num_of_subbands
            subband_signal(sub_band).SSVEPdata = zeros(n_ch,sig_len,d2,d1);
            subband_signal(sub_band).signal_template = zeros(n_ch,sig_len,d1);
            subband_signal(sub_band).testdata = zeros(n_ch,sig_len);
        end
    end

    for i=1:1:d1
        for j=1:1:d2
            y0=reshape(eeg(:,:,i,j),d3,d4);

            y = filtfilt(notchB, notchA, y0.'); %notch

            y = y.';

            for sub_band=1:num_of_subbands

                for ch_no=1:d3
                    tmp2=filtfilt(subband_signal(sub_band).bpB,subband_signal(sub_band).bpA,y(ch_no,:));
                    y_sb(ch_no,:) = tmp2(latencyDelay+1:latencyDelay+sig_len);
                end
                subband_signal(sub_band).SSVEPdata(:,:,j,i)=reshape(y_sb,d3,length(y_sb),1,1);

            end

        end
    end


    y0_test=test_data;

    y_test = filtfilt(notchB, notchA, y0_test.'); %notch
    y_test = y_test.';
    for sub_band=1:num_of_subbands

        for ch_no=1:d3
            tmp2_test=filtfilt(subband_signal(sub_band).bpB,subband_signal(sub_band).bpA,y_test(ch_no,:));
            y_sb_test(ch_no,:) = tmp2_test(latencyDelay+1:latencyDelay+sig_len);
        end

        subband_signal(sub_band).testdata(:,:)=reshape(y_sb_test,d3,length(y_sb_test));
    end

    clear eeg
    %% Initialization

    for sub_band=1:num_of_subbands
        subband_signal(sub_band).SSVEPdata=subband_signal(sub_band).SSVEPdata(:,:,:,target_order); % To sort the orders of the data as 8.0, 8.2, 8.4, ..., 15.8 Hz
    end

    FB_coef=FB_coef0'*ones(1,n_sti);
    n_correct=zeros(1,7); % Count how many correct detection



    for i=1:no_of_class

        for k=1:num_of_subbands

            subband_signal(k).signal_template(:,:,i)=mean(subband_signal(k).SSVEPdata(:,:,:,i),3);

        end

    end


    % for run_test=1:length(idx_testdata)
    % run_test=1;
    %             for tw_length=1:length(TW)
    clear Xa Xa_train
    %                 sig_len=TW_p(tw_length);
    test_signal=zeros(d3,sig_len);
    fprintf('Testing TW %fs',TW(tw_length));

    % TDCA training
    if enable_bit(5)==1
        for sub_band=1:num_of_subbands
            for j=1:no_of_class
                Ref=ref_signal_nh(sti_f(j),Fs,0,sig_len,num_of_harmonics);
                [Q_ref1,R_ref1]=qr(Ref',0);
                P=Q_ref1*Q_ref1';
                for train_no=1:d2
                    traindata_1a=[];
                    for dn=1:num_of_delay
                        traindata=reshape(subband_signal(sub_band).SSVEPdata(:,[dn:sig_len],train_no,j),d3,sig_len-dn+1);
                        if (is_center_std==1)
                            traindata=traindata-mean(traindata,2)*ones(1,length(traindata));
                            traindata=traindata./(std(traindata')'*ones(1,length(traindata)));
                        end
                        traindata_1a=[traindata_1a;[traindata zeros(length(ch_used),dn-1)]];
                    end
                    traindata_1a_P=traindata_1a*P;
                    if (is_center_std==1)
                        traindata_1a_P=traindata_1a_P-mean(traindata_1a_P,2)*ones(1,length(traindata_1a_P));
                        traindata_1a_P=traindata_1a_P./(std(traindata_1a_P')'*ones(1,length(traindata_1a_P)));
                    end
                    Xa(:,:,train_no,j)=[traindata_1a traindata_1a_P];
                end
                Xa_train(:,:,j,sub_band)=mean(Xa(:,:,:,j),3);
            end

            Sb=zeros(num_of_delay*d3);
            Sw=zeros(num_of_delay*d3);
            for j=1:no_of_class

                for train_no=1:d2
                    X_tmp=Xa(:,:,train_no,j);

                    Sw=Sw+X_tmp*X_tmp'/d2;
                end

                tmp=mean(Xa(:,:,:,j),3)-mean(mean(Xa,4),3);
                Sb=Sb+tmp*tmp'/no_of_class;
            end

            [eig_v1,eig_d1]=eig(Sw\Sb);
            [eig_val,sort_idx]=sort(diag(eig_d1),'descend');
            eig_vec=eig_v1(:,sort_idx(1:num_of_k));
            tdca_W(:,:,sub_band)=eig_vec;
        end
    end


    i=1;


    for sub_band=1:num_of_subbands
        test_signal=subband_signal(sub_band).testdata(:,1:TW_p(tw_length));%这里替换test_signal
        if (is_center_std==1)
            test_signal=test_signal-mean(test_signal,2)*ones(1,length(test_signal));
            test_signal=test_signal./(std(test_signal')'*ones(1,length(test_signal)));
        end
        for j=1:no_of_class
            template=subband_signal(sub_band).signal_template(:,[1:sig_len],j);
            if (is_center_std==1)
                template=template-mean(template,2)*ones(1,length(template));
                template=template./(std(template')'*ones(1,length(template)));
            end

            % Generate the sine-cosine reference signal
            ref1=ref_signal_nh(sti_f(j),Fs,pha_val(j),sig_len,num_of_harmonics);
            % ================ eCCA ===============
            if (enable_bit(1)==1)
                [ecca_r1,CR(sub_band,j),itR(sub_band,j),CCAR(sub_band,j)]=extendedCCA(test_signal,ref1,template,num_of_r);
            else
                CR(sub_band,j)=0;
                itR(sub_band,j)=0;
                CCAR(sub_band,j)=0;
            end

            % =============== ms-eCCA ===============
            if (enable_bit(2)==1)
                if (i==1)
                    % find the indices of neighboring templates
                    d0=floor(num_of_signal_templates/2);
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
                    for n_temp=1:num_of_signal_templates
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
                    [Wx1,Wy1,cr1]=canoncorr(mscca_template,mscca_ref(:,1:end));
                    spatial_filter1(sub_band,j).wx1=Wx1(:,1)';
                    spatial_filter1(sub_band,j).wy1=Wy1(:,1)';

                end


                cr1=corrcoef((spatial_filter1(sub_band,j).wx1*test_signal)',(spatial_filter1(sub_band,j).wy1*ref1)');
                cr2=corrcoef((spatial_filter1(sub_band,j).wx1*test_signal)',(spatial_filter1(sub_band,j).wx1*template)');
                %
                msccaR(sub_band,j)=sign(cr1(1,2))*cr1(1,2)^2+sign(cr2(1,2))*cr2(1,2)^2;
            else
                msccaR(sub_band,j)=0;
            end
            %===============eTRCA==================
            if (enable_bit(3)==1)
                if (num_of_trials==1)
                    % num_of_trials cannot be less than 2
                    % in TRCA
                    TRCAR(sub_band,j)=0;
                else
                    if ((i==1) && (j==1))
                        W_eTRCA(sub_band).val=[];
                        for jj=1:no_of_class
                            trca_X2=[];
                            trca_X1=zeros(d3,sig_len);
                            for tr=1:d2
                                X0=reshape(subband_signal(sub_band).SSVEPdata(:,1:sig_len,tr,jj),d3,sig_len);
                                if (is_center_std==1)
                                    X0=X0-mean(X0,2)*ones(1,length(X0));
                                    X0=X0./(std(X0')'*ones(1,length(X0)));
                                end
                                trca_X1=trca_X1+X0;
                                trca_X2=[trca_X2;X0'];
                            end
                            S=trca_X1*trca_X1'-trca_X2'*trca_X2;
                            Q=trca_X2'*trca_X2;
                            [eig_v1,eig_d1]=eig(Q\S);
                            [eig_val,sort_idx]=sort(diag(eig_d1),'descend');
                            eig_vec=eig_v1(:,sort_idx);
                            W_eTRCA(sub_band).val=[W_eTRCA(sub_band).val; eig_vec(:,1)'];
                        end
                    end

                    cr1=corrcoef(W_eTRCA(sub_band).val*test_signal,W_eTRCA(sub_band).val*template);
                    TRCAR(sub_band,j)=cr1(1,2);
                end
            else
                TRCAR(sub_band,j)=0;
            end
            %===============ms-eTRCA==================
            if (enable_bit(4)==1)
                if (num_of_trials==1)
                    % num_of_trials cannot be less than 2
                    % in eTRCA
                    MSTRCAR(sub_band,j)=0;
                else
                    if ((i==1) && (j==1))
                        W_msTRCA(sub_band).val=[];
                        for my_j=1:no_of_class
                            d0=floor(num_of_signal_templates2/2);
                            if my_j<=d0
                                template_st=1;
                                template_ed=num_of_signal_templates2;
                            elseif ((my_j>d0) && my_j<(d1-d0+1))
                                template_st=my_j-d0;
                                template_ed=my_j+(num_of_signal_templates2-d0-1);
                            else
                                template_st=(d1-num_of_signal_templates2+1);
                                template_ed=d1;
                            end
                            template_seq=[template_st:template_ed];
                            mstrca_X1=[];
                            mstrca_X2=[];

                            for n_temp=1:num_of_signal_templates2
                                jj=template_seq(n_temp);
                                trca_X2=[];
                                trca_X1=zeros(d3,sig_len);
                                template2=zeros(d3,sig_len);

                                for tr=1:d2
                                    X0=reshape(subband_signal(sub_band).SSVEPdata(:,1:sig_len,tr,jj),d3,sig_len);
                                    if (is_center_std==1)
                                        X0=X0-mean(X0,2)*ones(1,length(X0));
                                        X0=X0./(std(X0')'*ones(1,length(X0)));
                                    end
                                    trca_X2=[trca_X2;X0'];
                                    trca_X1=trca_X1+X0;
                                end
                                mstrca_X1=[mstrca_X1 trca_X1];
                                mstrca_X2=[mstrca_X2 trca_X2'];
                            end
                            S=mstrca_X1*mstrca_X1'-mstrca_X2*mstrca_X2';
                            Q=mstrca_X2*mstrca_X2';
                            [eig_v1,eig_d1]=eig(Q\S);
                            [eig_val,sort_idx]=sort(diag(eig_d1),'descend');
                            eig_vec=eig_v1(:,sort_idx);
                            W_msTRCA(sub_band).val=[W_msTRCA(sub_band).val; eig_vec(:,1)'];
                        end
                    end
                    cr1=corrcoef(W_msTRCA(sub_band).val*test_signal,W_msTRCA(sub_band).val*template);
                    MSTRCAR(sub_band,j)=cr1(1,2);

                    if (enable_bit(2)==1)
                        cr2=corrcoef((spatial_filter1(sub_band,j).wx1*test_signal)',(spatial_filter1(sub_band,j).wy1*ref1)');
                        MSCCATRCAR(sub_band,j)=sign(cr1(1,2))*cr1(1,2)^2+sign(cr2(1,2))*cr2(1,2)^2;
                    else
                        MSCCATRCAR(sub_band,j)=0;
                    end
                end
            else
                MSTRCAR(sub_band,j)=0;
                MSCCATRCAR(sub_band,j)=0;
            end
            %===============TDCA==================
            if (enable_bit(5)==1)
                if (num_of_trials==1)
                    % num_of_trials cannot be less than 2
                    TDCAR(sub_band,j)=0;
                else
                    test_signal_1a=[];
                    for dn=1:num_of_delay
                        z=[test_signal(:,dn:end) zeros(length(ch_used),dn-1)];
                        test_signal_1a=[test_signal_1a;z];
                    end

                    Ref=ref_signal_nh(sti_f(j),Fs,0,sig_len,num_of_harmonics);
                    [Q_ref1,R_ref1]=qr(Ref',0);
                    P=Q_ref1*Q_ref1';
                    test_signal_1a_P=test_signal_1a*P;
                    Xb=[test_signal_1a test_signal_1a_P];
                    W=tdca_W(:,:,sub_band);
                    TDCAR(sub_band,j)=corr2(W'*Xb,W'*Xa_train(:,:,j,sub_band));

                end
            else
                TDCAR(sub_band,j)=0;
            end

        end

    end

    CCAR1=sum((CCAR).*FB_coef,1);
    CR1=sum((CR).*FB_coef,1);
    msccaR1=sum((msccaR).*FB_coef,1);
    TRCAR1=sum((TRCAR).*FB_coef,1);
    MSTRCAR1=sum((MSTRCAR).*FB_coef,1);
    MSCCATRCAR1=sum((MSCCATRCAR).*FB_coef,1);
    TDCAR1=sum((TDCAR).*FB_coef,1);
    if enable_bit(1)==1
        [~,idx]=max(CCAR1);
        n_correct(1)=idx;

        [~,idx]=max(CR1);
        n_correct(2)=idx;
    end
    if enable_bit(2)==1
        [~,idx]=max(msccaR1);
        n_correct(3)=idx;
    end
    if enable_bit(3)==1
        [~,idx]=max(TRCAR1);
        n_correct(4)=idx;
    end
    if enable_bit(4)==1
        [~,idx]=max(MSTRCAR1);
        n_correct(5)=idx;
    end
    if enable_bit(2)==1
        [~,idx]=max(MSCCATRCAR1);
        n_correct(6)=idx;
    end
    if enable_bit(5)==1
        [~,idx]=max(TDCAR1);
        n_correct(7)=idx;
    end
    n_correct
    % end
    %             end
    % end



    %% Save results
    toc

    % all_sub_acc(:,:,i)
    % i= 1: CCA
    % i= 2: eCCA
    % i= 3: ms-eCCA
    % i= 4: eTRCA
    % i= 5: ms-eTRCA
    % i= 6: ms-eCCA + ms-eTRCA
    % i= 7: TDCA


end
