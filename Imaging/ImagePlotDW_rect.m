%% reconstruct the 2d image from a 1d array points according to the 2D index
% information
IPD_FrameData0 = zeros(length(pixelMap.zaxis), length(pixelMap.xaxis));  % initialize the 2d image in terms of its sizes
IPD_FrameData = zeros(length(pixelMap.zaxis), length(pixelMap.xaxis),size(DataFA,4)); % initialize the 2d image and frame no size
IPD_index = pixelMap.IndexBeamforming;  % 2d index information 
for i = 1:size(DataFA,4)
    IPD_FrameData0(IPD_index) = DataFormed(:,i);
    IPD_FrameData(:,:,i) = IPD_FrameData0;
end    

DataBFormed = IPD_FrameData;

%%
FrameData = abs(DataBFormed);
magd = max(abs(FrameData(:)));
IPD_F(size(FrameData,3)) = struct('cdata',[],'colormap',[]);
% figure('Position', [100, 100, 512, 512]);
for i = 1:size(FrameData,3)
    figure;
%     figure('Position', [0 0 212 212])
    IPD_im1 = FrameData(:,:,i);
    if 1
        LogData = 20*log10(IPD_im1/magd);
        imagesc(pixelMap.xaxis*1000,pixelMap.zaxis*1000,LogData,[-57,0]);
        colorbar;
    else
        im2=20*log10(IPD_im1/magd)+50;
        im2(im2<0)=0;
        im3=5.1*im2;im3(im3<26.7)=0;
        image(pixelMap.xaxis*1000,pixelMap.zaxis*1000,im3);colorbar;
    end
    set(gca,'YDir','normal');
    axis equal tight ;
    axis off ; 
    colorbar off;%隐藏坐标轴 
    colormap gray ;
%     title(['Frame No:',num2str(i)]);
    xlabel('mm');
    IPD_F(i) = getframe(gcf);
end

