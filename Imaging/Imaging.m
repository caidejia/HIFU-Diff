% this file is to beamform the data from the conventional focused transmission    该文件用于对来自传统聚焦传输的数据进行波束成形
% by Xiaowei Zhou, 2020.09.18 at Haifu
clear all;clc;;
%% the necessary parameters about the beamforming modes 
Option.ImagingMode = 5; % 1: LA PW, 2: LA DW, 3: LA FUS, 4: CA FUS, 5: CA DW
Option.BF_Matrix = 0; % 1 for using the DAS beamforming matrix, 0 for normal DAS interpolation                     DAS（delay and sum 延迟与求和）1表示DAS波束成形矩阵，0 表示正常的 DAS 插值
Option.IQdemo = 0;  % 1 for using the IQ demodulated signal for the migration method                               使用IQ解调信号进行迁移的方法
Option.Hilbt = 1;  % 1 for using the IQ demodulated signal for the migration method                                    信号分析
Option.ROIChoose = 0;% 1 for the whole scanning region, 2 for a choosen region                                        1，表示整个区域   2.选定区域
Option.filter = 0;  % 1 for using a bandpass filter according to the transducer used, 0 for not using any filte rs      1.表示带通。0，表示不用任何滤波器
Option.savetype = 1; % 1 for rf data, 2 for iq data or analytical data from hilbert transform                          1.表示直接的RF信号，2，表示经过希尔伯特变换的复数IQ信号
Option.Polar = 1; % 1 for using polar grid for the divering wave DAS, 2 for using the rectangle grid                 1，极坐标  2.矩阵坐标
Option.Sensitivity = 1;% 1 for using the relative sensitivity of the probe element in Recon
Option.fnumber = 0;    % the selected f# value for the receiving beamforming                                          ？？求和的常数值
Option.sumMode = 'SUM';   % SUM, CF, GCF, MV, MSUM, SLSC.                                                                应该是选择不同模式吧
addpath(genpath('./functions/'));
%% load the data
path_setting = 'Imaging_setting.mat';
path_data = '../store_pre/output1.mat';
load(path_setting);
load(path_data);

%% args set
isSave = true;
ROISave = false;
mode = 3; % 1 LABEL 2 DATA 3 PREDICT
init_angle = 15; % If single-angle imaging is required, it is necessary to set the corresponding angle index.
savepath = './';
savename = 'divergingWave';
savepath_ROI = strcat(savepath,savename," ROI");
if mode == 1 % Load the variable names from the MAT data.
    DataFA = Label;
elseif mode == 2
    DataFA = Data;
else
    DataFA = predict;
end

%% necessary general parameters to be defined in the following format strictly    
param.Interplo_Type = 'linear';  % could be 'nearest' or 'linear' 
param.c = 1540;  % speed of sound in m/s        
param.ImageDepth = UserSet.depth2; % in m
param.fs = Receive(1).decimSampleRate*1e6;  % Sampling frequency in Hz  
% param.fs = 100e6;
param.fc = Trans.frequency*1e6;  % central frequency of the probe in Hz
param.Tx = UserSet.tx_freq*1e6;   % the transmitted frequency in Hz, this is optional since it is only required for the IQ demolution and it can be estimated.
param.pitch = Trans.spacingMm/1000;  % m
% param.pitch = Trans.Ele_pitch;  % the pitch size of the probe in m

param.PixelSizeX = 250E-6;  param.PixelSizeZ = 250E-6;   % The pixel size defined for the final image to be beamformed, in m
% param.PixelSizeX = 385E-6;  param.PixelSizeZ = 281E-6;   % The pixel size defined for the final image to be beamformed, in m
param.na = size(DataFA,3);
% UserSet.angle_range=0;
param.fnumber = Option.fnumber;      % the f# for the receiving beamforming
% param.RXangle = 0; % the angle used for the receiving beamforming with split aperture for Vector Doppler in degree
% param.na = UserSet.angle_num;  % number of compounding angles
param.AngleRange = UserSet.angle_range;  % the angle range of the compounded tranmission in degrees
% param.t0 = Trans.lensCorrection*2/param.fc;  % the time of lens correction of the probe in s
param.t0 = Trans.lensCorrection*2/param.c/1000;  % the time of lens correction of the probe in s
% param.t0 = 0;
param.Nelements = size(DataFA,2);   % number of elements for receiving in the imaging
param.ElePositionX = Trans.ElementPos(:,1)/1000; % transducer element positions in lateral direction m
param.ElePositionZ = Trans.ElementPos(:,3)/1000; % transducer element positions in axial direction m
if Option.filter == 1   % bandpass filter
    param.lowcutoff = 5e6;   % in Hz, the lower cutoff for the bandpass filter
    param.highcutoff = 11e6;   % in Hz, the lower cutoff for the bandpass filter
end
param.focus = -Trans.radiusMm/1000;   % the geometric focus,  in m
param.OpenAngleHalf = -Trans.ElementPos(1,4);
param.polar = Option.Polar;  % to indicate that the polar system will be used in the DAS algorithm DasMigrate_zhou.m
if Option.ROIChoose == 2   % define the region to be beamformed
    param.Lateral1 = -70/1000;
    param.Lateral2 = 70/1000;   % in m
    %     param.Lateral1 = param.ElePositionX(1)-50/1000;
    %     param.Lateral2 = param.ElePositionX(end)+50/1000;   % in m
    param.Depth1 = 90/1000;
    param.Depth2 = 140/1000; % in m
else
    param.Depth1 = 0/1000;  param.Depth2 = param.ImageDepth;
    param.Lateral1 = -(param.Depth2-param.focus)*sin(param.OpenAngleHalf);
    param.Lateral2 = -param.Lateral1;
end
param.mode = Option.ImagingMode;
param.sumMode = Option.sumMode;
%%%%%% normally do not change anything beyond this line
% param.na = 1;
if param.na == 1
    a = linspace(-param.AngleRange/2, param.AngleRange/2, UserSet.angle_num);
    param.TXangle = a(init_angle);   % the angles used for the compounded tranmission in degrees
else
    param.TXangle = linspace(-param.AngleRange/2, param.AngleRange/2, param.na);  % degree
end
%-- transducer element's relative sensitivity (directivity)
if Option.Sensitivity == 1
    param.Senscutoff = UserSet.SensCutOff; % Sensitivity cut off value in [0,1]
    % Set element sensitivity function (101 weighting values from -pi/2 to pi/2).
    Theta = (-pi/2:pi/100:pi/2);
    Theta(51) = 0.0000001; % set to almost zero to avoid divide by zero.
    if ~isfield(Trans,'ElementSens')
        ElementWidth = Trans.elementWidth/1000;  % the element width of the probe in m
        ElementWidthWl = ElementWidth * param.fc/param.c;
        param.ElementSens = abs(cos(Theta).*(sin(ElementWidthWl*pi*sin(Theta))./(ElementWidthWl*pi*sin(Theta))));
    else
        param.ElementSens = Trans.ElementSens;
    end
    [~,AngleIndex] = min(abs(param.ElementSens  - param.Senscutoff));
    param.SenscutoffAngle = abs(Theta(AngleIndex)); % Sensitivity cut off Angle absolute value in radian
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    No        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   need       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    to        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  change      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  anything    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   below      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    this      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   point      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% define the pixel map that is required to be beamformed, the generated structure is pixelMap


PixelMapGenerate_DW_rect


%% filter the data according to the bandwidth of the probe  
if Option.filter == 1
    Wn = [param.lowcutoff,param.highcutoff]/(param.fs/2);
    [B,A] = butter(5,Wn);
    DataFA1 = zeros(size(DataFA));
    for i=1:size(DataFA,4)
        DataFA1(:,:,:,i) = filtfilt(B,A,DataFA(:,:,:,i));
    end
    DataFA = DataFA1;
    clear DataFA1
end
%% do the beamforming by calling the Migration.m, which will generate a beamformed matrix called "DataFormed" WITH the size [length(pixelMap.zaxis), length(pixelMap.xaxis), frames]
tic
DataFormed = Migration_SuiteXZ(double(DataFA),param,pixelMap,Option);
toc

%% plot the images and save the data
ImagePlotDW_rect
if isSave
    %         print('-dtiff', [savepath,surfix], '-r600')
    saveas(gcf, [savepath,savename], 'png');
end
if ROISave
    ImageROI
    print('-dtiff', savepath_ROI, '-r600')
    %             saveas(gcf, savepath_ROI, 'tif');
end



