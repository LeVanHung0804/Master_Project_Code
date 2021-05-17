%10/02/2020
%This function to get 6 cycles of ppg wave from filtered ppg signal
%Affter get 6 cycles, I continue resampling it to vector with 3000 element
  
function result = get_2_ppg_vpg_apg_cycles(ppgIn,No)
a = [1:1:3500];
flag = 0;

%find max peak
[max_peak,max_index] = findpeaks(ppgIn,a,'MinPeakHeight',0,'MinPeakDistance',250);
%find min peak
[min_peak,min_index] = findpeaks(-ppgIn,a,'MinPeakHeight',0,'MinPeakDistance',250);
%check the accuracy of those peaks
%finded peaks are true if distrubtution of peaks are: ...min-max-min-max...
if abs(length(min_index)-length(max_index)) ~= 1 && length(min_index)~=length(max_index) 
    plot(a,ppgIn);
    hold on;
    plot(max_index,max_peak,'r o');
    hold on;
    plot(min_index,-min_peak,'r o');
    warning('finded peaks are not true,length!=length',No)
    flag =1;
    
end
if length(min_index)>length(max_index) 
    num_cycle = length(max_index)
else
    num_cycle = length(min_index)
end
if min_index(1)>max_index(1)
    for i=1:num_cycle
        if min_index(i)<max_index(i)
            plot(a,ppgIn);
            hold on;
            plot(max_index,max_peak,'r o');
            hold on;
            plot(min_index,-min_peak,'r o');
            warning('finded peaks are not true,the order is incorrect',No)
            flag = 1;
        end
    end
else
    for i=1:num_cycle
        if min_index(i)>max_index(i)
            plot(a,ppgIn);
            hold on;
            plot(max_index,max_peak,'r o');
            hold on;
            plot(min_index,-min_peak,'r o');
            warning('finded peaks are not true,the order is incorrect',No)
            flag = 1;
        end
    end
end
%split ppg and pick num_cycle
if num_cycle<6
    plot(a,ppgIn);
    hold on;
    plot(max_index,max_peak,'r o');
    hold on;
    plot(min_index,-min_peak,'r o');
    warning('the cycle is too big',No)
    flag = 1;
end

one_cycle_ppg = ppgIn(1,min_index(1):min_index(3));

originalFs = length(one_cycle_ppg);

input2 = originalFs;

SAMPLE_FREQUENCY = 500
desiredFs = 1000;  %1000
[p,q] = rat((desiredFs+6) / originalFs);

x = 1/SAMPLE_FREQUENCY:1/SAMPLE_FREQUENCY:(1/SAMPLE_FREQUENCY)*originalFs;
y = one_cycle_ppg;
dy = derivative(y,x);
ddy = derivative(dy,x);

y = resample(y,p,q);
dy = resample(dy,p,q);
ddy = resample(ddy,p,q);

y(1:3)=[];
dy(1:3) = [];
ddy(1:3) = [];

if length(y)>desiredFs
    y(desiredFs+1:length(y))=[];
    dy(desiredFs+1:length(dy))=[];
    ddy(desiredFs+1:length(ddy))=[];
end
if length(y)<desiredFs
    y(length(y):desiredFs)=y(length(y));
    dy(length(dy):desiredFs)=dy(length(dy));
    ddy(length(ddy):desiredFs)=ddy(length(ddy));
end


result = [y,dy,ddy,flag,input2];

end

