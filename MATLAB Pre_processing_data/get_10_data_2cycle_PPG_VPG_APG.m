%=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:
% Filter data from raw data of Lab
% Expect data shape after filter: (3000,1)
% Hz = 500 Hz:
% Created by Hungle
% Base on HUY

clear all;
close all;  
clc;
fclose all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%DEFINE PARA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data
WAVE_DATA_PATH = 'D:\lab_file\BP_Hungle\data\data_DrLEE_ranged\wave_data.xlsx';

raw_data=xlsread(WAVE_DATA_PATH);

OUTPUT_DATA_PATH = 'D:\lab_file\BP_Hungle\data\data_DrLEE_ranged\output.xlsx';

output_data = xlsread(OUTPUT_DATA_PATH);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%split data

[row,column] = size(raw_data);
new_column = column;
%index = zeros(column,1);
output = zeros(column,2);

for i = 1:length(output_data)
        output(3*(i-1)+1,1) = output_data(i,1);
        output(3*(i-1)+2,1) = output_data(i,3);
        output(3*(i-1)+3,1) = output_data(i,5);
    
        output(3*(i-1)+1,2) = output_data(i,2);
        output(3*(i-1)+2,2) = output_data(i,4);
        output(3*(i-1)+3,2) = output_data(i,6);
end

for i=1:column
    if raw_data(1,i) == 1
        new_column = new_column - 1;
    end
end

new_raw_data = zeros(row,new_column);
new_output = zeros(new_column,2);
j = 0;
for i = 1:column
    if raw_data(1,i) == 1
        ;
    else
        j = j+1;
        new_raw_data(:,j) = raw_data(:,i);
        new_output(j,:) = output(i,:);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%change frequency from 1000 to 500
new_raw_data(1:4,:)=[];
%changed_raw_data = zeros(floor(length(new_raw_data)/2),new_column);
for i=1:floor(length(new_raw_data)/2)
    changed_raw_data(i,:) = new_raw_data(2*i,:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%filter data
% signal                         by using filter_proc
samplerate=500; % in Hz
passband=[0.6 6];
changed_raw_data(1:2*samplerate,:)=[];
for i=1:new_column
    filtered_data(:,i)=filter_proc(changed_raw_data(:,i)',samplerate,passband,3)';
end

% save data to dataAll structure
for i=1:new_column
    disp(['loading rawStream from column ',i]);
    dataAll(i).ppgFiltered = filtered_data(:,i)';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% by using filter filter3.mat
%FILTER_NAME = 'filter3'
%load(FILTER_NAME);
%eval(['f=',FILTER_NAME,';']);    %FILTER_NAME = 'filter3'
%f = readFilter();
%for i=1:length(changed_raw_data)
%    filteredd_data(:,i)=filter(f, 1, changed_raw_data(:,i));
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%check data by PSD
% PSD
SAMPLE_FREQUENCY = 500;
WINDOW_TIME = 7; % unit: sec
MOVING_STEP = 0.5; % unit: sec

% high psd extended
NUMBER_OF_EXTENDED = 10;

% set variable
stepSize   = MOVING_STEP * SAMPLE_FREQUENCY; % unit: numbers of samples
windowSize = WINDOW_TIME * SAMPLE_FREQUENCY; % unit: numbers of samples

for i=1:new_column
    % display message (this loop may take long time)
    disp(['calculating psd of column ',i]);
    
    % initialize
    ppgAll   = dataAll(i).ppgFiltered;
    psdArray = -1.*ones(size(ppgAll));
    
    % loop for caculating psd value
    for j=1:stepSize:length(psdArray)
        % skip head and tail
        if (j==1)||(j+windowSize-1>=length(psdArray))
            continue;
        end
        
        % get the window of ppg
        ppgWindow = ppgAll(j:j+windowSize-1);
        
        % get psd
        psd = getPsdOfPpg_20180915(ppgWindow); % with high resolution version of fft
        psdArray(j) = psd;
    end
    
    % save the psd Array
    dataAll(i).psdArray = psdArray;
end

for i=1:new_column
    % get the NUMBER_OF_EXTENDED highest psd index
    psdTemp = dataAll(i).psdArray;
    [sortValue, sortIndex] = sort(psdTemp,'descend');
    nanArray = isnan(sortValue);
    sortValue(nanArray) = [];
    sortIndex(nanArray) = [];
    
    % check if there is enough psd calculated
    % NOTE: this error happended if the ppg is not long enough or
    %       MOVING_STEP too large that no enough psd calculated
    if sortValue(NUMBER_OF_EXTENDED)==-1
        error(['No enough psd calculated in file ',num2str(i)]);
    end
    
    % save the index of the selected psd
    dataAll(i).psdSelected = sortIndex(1:NUMBER_OF_EXTENDED);
    dataAll(i).sortIndex = sortIndex;
    dataAll(i).sortValue = sortValue;
    dataAll(i).max_psd = sortValue(1);
    
    % save the index of the selected psd with phase shift corrected 
    %phaseShift = floor(length(f)/2);
    %correctedIndex = sortIndex(1:NUMBER_OF_EXTENDED) - phaseShift;
    %correctedIndex(correctedIndex<=1) = 1;% to prevent bug
    %dataAll(i).psdSelectedCorrected = correctedIndex;
end
%3500
for i=1:new_column
    selectedIdx = dataAll(i).psdSelected;
    dataAll(i).ppg_3000_1 = dataAll(i).ppgFiltered(selectedIdx(1):selectedIdx(1)+windowSize-1);
    dataAll(i).ppg_3000_2 = dataAll(i).ppgFiltered(selectedIdx(2):selectedIdx(2)+windowSize-1);
    dataAll(i).ppg_3000_3 = dataAll(i).ppgFiltered(selectedIdx(3):selectedIdx(3)+windowSize-1);
    dataAll(i).ppg_3000_4 = dataAll(i).ppgFiltered(selectedIdx(4):selectedIdx(4)+windowSize-1);
    dataAll(i).ppg_3000_5 = dataAll(i).ppgFiltered(selectedIdx(5):selectedIdx(5)+windowSize-1);
    dataAll(i).ppg_3000_6 = dataAll(i).ppgFiltered(selectedIdx(6):selectedIdx(6)+windowSize-1);
    dataAll(i).ppg_3000_7 = dataAll(i).ppgFiltered(selectedIdx(7):selectedIdx(7)+windowSize-1);
    dataAll(i).ppg_3000_8 = dataAll(i).ppgFiltered(selectedIdx(8):selectedIdx(8)+windowSize-1);
    dataAll(i).ppg_3000_9 = dataAll(i).ppgFiltered(selectedIdx(9):selectedIdx(9)+windowSize-1);
    dataAll(i).ppg_3000_10 = dataAll(i).ppgFiltered(selectedIdx(10):selectedIdx(10)+windowSize-1);
    dataAll(i).sbp = new_output(i,1);
    dataAll(i).dbp = new_output(i,2);
end

%check ppg signal with max and min psd value
%index_max = 1;
%index_min = 1;
%max_psd = 0;
%min_psd = 1;
%for i=1:new_column
%    if max_psd<dataAll(i).max_psd
%        max_psd = dataAll(i).max_psd;
%        index_max = i;
%    end
%    if min_psd>dataAll(i).max_psd
%        min_psd = dataAll(i).max_psd;
%        index_min = i;
%    end
%end

%max_psd_ppg = dataAll(index_max).ppg_3000;
%min_psd_ppg = dataAll(index_min).ppg_3000;

disp('filter process done!!!')
disp('saving to excel file-loanding')

desiredFs = 3000; %1000*3 = 3000
data_input = zeros(1,desiredFs);
data_output = zeros(1,2);
input_2 = zeros(1,1);
ok_psd = 0.35
for i =1:length(dataAll)
    ppgWindow = dataAll(i).ppg_3000_1;
    psd = getPsdOfPpg_20180915(ppgWindow);
    if psd>ok_psd
        result = get_2_ppg_vpg_apg_cycles(ppgWindow,i-1);
        if result(desiredFs+1) ~= 1
        data_input = [data_input;result(1:desiredFs)];
        data_output = [data_output;[dataAll(i).sbp,dataAll(i).dbp]];
        input_2 = [input_2;result(desiredFs+2)];
        end
    end
    ppgWindow = dataAll(i).ppg_3000_2;
    psd = getPsdOfPpg_20180915(ppgWindow);
    if psd>ok_psd
        result = get_2_ppg_vpg_apg_cycles(ppgWindow,i-1);
        if result(desiredFs+1) ~= 1
            data_input = [data_input;result(1:desiredFs)];
            data_output = [data_output;[dataAll(i).sbp,dataAll(i).dbp]];
            input_2 = [input_2;result(desiredFs+2)];
        end
    end
    ppgWindow = dataAll(i).ppg_3000_3;
    psd = getPsdOfPpg_20180915(ppgWindow);
    if psd>ok_psd
        result = get_2_ppg_vpg_apg_cycles(ppgWindow,i-1);
        if result(desiredFs+1) ~= 1
            data_input = [data_input;result(1:desiredFs)];
            data_output = [data_output;[dataAll(i).sbp,dataAll(i).dbp]];
            input_2 = [input_2;result(desiredFs+2)];
        end
    end
    ppgWindow = dataAll(i).ppg_3000_4;
    psd = getPsdOfPpg_20180915(ppgWindow);
    if psd>ok_psd
        result = get_2_ppg_vpg_apg_cycles(ppgWindow,i-1);
        if result(desiredFs+1) ~= 1
            data_input = [data_input;result(1:desiredFs)];
            data_output = [data_output;[dataAll(i).sbp,dataAll(i).dbp]];
            input_2 = [input_2;result(desiredFs+2)];
        end
    end
    ppgWindow = dataAll(i).ppg_3000_5;
    psd = getPsdOfPpg_20180915(ppgWindow);
    if psd>ok_psd
        result = get_2_ppg_vpg_apg_cycles(ppgWindow,i-1);
        if result(desiredFs+1) ~= 1
        data_input = [data_input;result(1:desiredFs)];
        data_output = [data_output;[dataAll(i).sbp,dataAll(i).dbp]];
        input_2 = [input_2;result(desiredFs+2)];
        end
    end
    ppgWindow = dataAll(i).ppg_3000_6;
    psd = getPsdOfPpg_20180915(ppgWindow);
    if psd>ok_psd
        result = get_2_ppg_vpg_apg_cycles(ppgWindow,i-1);
        if result(desiredFs+1) ~= 1
        data_input = [data_input;result(1:desiredFs)];
        data_output = [data_output;[dataAll(i).sbp,dataAll(i).dbp]];
        input_2 = [input_2;result(desiredFs+2)];
        end
    end
    ppgWindow = dataAll(i).ppg_3000_7;
    psd = getPsdOfPpg_20180915(ppgWindow);
    if psd>ok_psd
        result = get_2_ppg_vpg_apg_cycles(ppgWindow,i-1);
        if result(desiredFs+1) ~= 1
        data_input = [data_input;result(1:desiredFs)];
        data_output = [data_output;[dataAll(i).sbp,dataAll(i).dbp]];
        input_2 = [input_2;result(desiredFs+2)];
        end
    end
    ppgWindow = dataAll(i).ppg_3000_8;
    psd = getPsdOfPpg_20180915(ppgWindow);
    if psd>ok_psd
        result = get_2_ppg_vpg_apg_cycles(ppgWindow,i-1);
        if result(desiredFs+1) ~= 1
        data_input = [data_input;result(1:desiredFs)];
        data_output = [data_output;[dataAll(i).sbp,dataAll(i).dbp]];
        input_2 = [input_2;result(desiredFs+2)];
        end
    end
    ppgWindow = dataAll(i).ppg_3000_9;
    psd = getPsdOfPpg_20180915(ppgWindow);
    if psd>ok_psd
        result = get_2_ppg_vpg_apg_cycles(ppgWindow,i-1);
        if result(desiredFs+1) ~= 1
        data_input = [data_input;result(1:desiredFs)];
        data_output = [data_output;[dataAll(i).sbp,dataAll(i).dbp]];
        input_2 = [input_2;result(desiredFs+2)];
        end
    end
    ppgWindow = dataAll(i).ppg_3000_10;
    psd = getPsdOfPpg_20180915(ppgWindow);
    if psd>ok_psd
        result = get_2_ppg_vpg_apg_cycles(ppgWindow,i-1);
        if result(desiredFs+1) ~= 1
        data_input = [data_input;result(1:desiredFs)];
        data_output = [data_output;[dataAll(i).sbp,dataAll(i).dbp]];
        input_2 = [input_2;result(desiredFs+2)];
        end
    end
end

% tat ca cac gia tri cua song lon hon khong
% giam thieu rui do khi dung ham relu
% [m,n] = size(data_input)
% for i=2:m
%     temp = data_input(i,:);
%     min_temp = min(temp);
%     data_input(i,:) = data_input(i,:) - min_temp;
% end
input_PPG = data_input(:,1:1000);
input_VPG = data_input(:,1001:2000);
input_APG = data_input(:,2001:3000);
% tat ca cac gia tri cua song lon hon khong
% giam thieu rui do khi dung ham relu
[m,n] = size(input_PPG);
for i=2:m
    temp = input_PPG(i,:);
    min_temp = min(temp);
    input_PPG(i,:) = input_PPG(i,:) - min_temp;
end

[m,n] = size(input_VPG);
for i=2:m
    temp = input_VPG(i,:);
    min_temp = min(temp);
    input_VPG(i,:) = input_VPG(i,:) - min_temp;
end

[m,n] = size(input_APG);
for i=2:m
    temp = input_APG(i,:);
    min_temp = min(temp);
    input_APG(i,:) = input_APG(i,:) - min_temp;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


max_PPG = max(input_PPG(:));
max_VPG = max(input_VPG(:));
max_APG = max(input_APG(:));

input_PPG = input_PPG/max_PPG;
input_VPG = input_VPG/max_VPG;
input_APG = input_APG/max_APG;

data_input = [input_PPG,input_VPG,input_APG];

xlswrite('input1_2000_10_PPG_VPG_APG_2cycles.xlsx',data_input)
xlswrite('output_2000_10_PPG_VPG_APG_2cycles',data_output)
xlswrite('input2_2000_10_PPG_VPG_APG_2cycles.xlsx',input_2)
disp('saving to excel file-DONE!!!')
%%%%%%%%%%%%%%%%test model
