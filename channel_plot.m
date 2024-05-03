% 生成随机数据（这里假设你已经有了你的实际数据）
data = data(1:24,:,1,1); % 替换成你的实际数据

% 归一化处理每个通道的数据
normalized_data = zeros(size(data));
for i = 1:size(data, 1)
    normalized_data(i, :) = normalize(data(i, :));
end

% 绘制波形图
figure;
hold on;
for i = 1:size(normalized_data, 1)
    plot(normalized_data(i, :) + 2*i); % 在 Y 轴上逐渐偏移，以便分散排列
end
hold off;

% 设置图形属性
xlabel('Time');
ylabel('Channel');
title('Waveforms for 25 Channels');
grid on;
