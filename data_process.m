%%用eeglab截取[0 2]的数据，
% 假设你的原始数据存储在一个名为 original_data 的 63*2000*120 的数组中
[fileFullNames, filepath] = uigetfile('*.edf','选择脑电数据','C:\Users\94322\Desktop\matlab_script\ssvep_data\SSVEP_data_2024');
EEG = pop_biosig( [filepath,'\',fileFullNames]);
EEG = pop_epoch( EEG, {  }, [-0.5         2], 'newname', 'EDF file epochs', 'epochinfo', 'yes');
EEG = pop_rmbase( EEG, [-500 0] ,[]);

% 创建一个新数组来存储整理后的数据
target=9;
block=6;
new_data = zeros(size(EEG.data,1),size(EEG.data,2), target, block);
for i=1:block
    data=EEG.data(:, :, (target*(i-1)+1):target*i);
    new_data(:, :, :, i) = data;
end
data=new_data;
save S1.mat data
% 现在 new_data 就包含了整理后的数据，每个 40*3 的切片代表了相应范围的原始数据
