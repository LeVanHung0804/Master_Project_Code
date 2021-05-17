%=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:
%=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:
% getPsdOfPpg_20180915.m
%=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:
% Last update: 20180915
%=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:
% Feature:
% 1. aclculate the PSD of the input ppg signal
% 2.
%=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:
% input
% ppg : ppg signal
%
% output
% psd
%=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:
%=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:=:
function result = getPsdOfPpg_20180915(ppgIn)

%==========================================================================
% parse the input parameter of function
%==========================================================================
ppg    = ppgIn;

%--------------------------------------------------------------------------
%                         CONFIGURATION PARSING
%--------------------------------------------------------------------------
% token
TOKEN = '';

% signal
SAMPLE_FREQUENCY = 500;

% fft
% NOTE: size=0 means no padding, size=0.2 means 20% padding at head and
%       tail, remember to detrend after padding
SIZE_PADDING = 0;
HIGHER_RESOLUTION = 1;  % resolution is: nfft = 2^(nextpow2(len)+RESOLUTION);

% hr
MAX_HR = 2; % unit: Hz
MIN_HR = 0.8; % unit: Hz

% psd
SPECTRUM_POWER_SUM_RANGE = 1;    % unit: +-1 points of fft stamp
SPECTRUM_HARMONIC_RANGE  = 4;    % unit: +-5 points of fft stamp
SPECTRUM_HARMONIC_USED   = 3;     % use 3 harmonic for psd

%--------------------------------------------------------------------------
%                           END CONFIGURATION
%--------------------------------------------------------------------------

%==========================================================================
% check token
%==========================================================================
% if ~strcmp(TOKEN,'setPsdOfPpgConfig_20180709')
%     error('getPsdOfPpg: check the token before continue')
% end

%==========================================================================
% do padding
%==========================================================================
% NOTE: need detrend after padding
%==========================================================================
if SIZE_PADDING~=0
    error('getPsdOfPpg: padding is not finished yet')
end

%==========================================================================
% get psd
%==========================================================================
% fft
len = length(ppg);
nfft = 2^(nextpow2(len)+HIGHER_RESOLUTION);
spectrum = fft(ppg,nfft)/len;
spectrum = 2*abs(spectrum(1:nfft/2+1));
spectrumStamp = SAMPLE_FREQUENCY/2*linspace(0,1,nfft/2+1);

if SPECTRUM_HARMONIC_USED~=3
    error('getPsdOfPpg: only SPECTRUM_HARMONIC_USED = 3 is available now');
end

% find HR peaks (1-st harmonic)
temp = spectrum;
temp(spectrumStamp<MIN_HR | spectrumStamp>MAX_HR) = -inf;
[~,maxIdx] = max(temp);

hr1Index = maxIdx;
hr1Freq = spectrumStamp(hr1Index);
hr1Power = sum(spectrum(hr1Index-SPECTRUM_POWER_SUM_RANGE:hr1Index+SPECTRUM_POWER_SUM_RANGE));

% 2-nd harmonic
temp = spectrum;
temp([1:hr1Index*2-SPECTRUM_HARMONIC_RANGE , hr1Index*2+SPECTRUM_HARMONIC_RANGE:end]) = -inf;
[~,maxIdx] = max(temp);

hr2Index = maxIdx;
hr2Freq = spectrumStamp(hr2Index);
hr2Power = sum(spectrum(hr2Index-SPECTRUM_POWER_SUM_RANGE:hr2Index+SPECTRUM_POWER_SUM_RANGE));

% 3-rd harmonic
temp = spectrum;
temp([1:hr1Index*3-SPECTRUM_HARMONIC_RANGE , hr1Index*3+SPECTRUM_HARMONIC_RANGE:end]) = -inf;
[~,maxIdx] = max(temp);

hr3Index = maxIdx;
hr3Freq = spectrumStamp(hr3Index);
hr3Power = sum(spectrum(hr3Index-SPECTRUM_POWER_SUM_RANGE:hr3Index+SPECTRUM_POWER_SUM_RANGE));

% calculate PSD
psd = (hr1Power+hr2Power+hr3Power)/sum(spectrum);

%==========================================================================
% set the output result of this function
%==========================================================================
result = psd;

end