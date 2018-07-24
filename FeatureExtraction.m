function [t state X] = FeatureExtraction (edf_path , hypnogram_path)
%------------------------------------------------------------------
hypnogram = AnnotExtract(hypnogram_path) ; % Loading the hypnogram files
hypnogram_time = hypnogram (1,:);
hypnogram_state = hypnogram (2,:);
%------------------------------------------------------------------
edfheader = edfread (edf_path,'assignToVariables',true) ; % Loading the edf files
%------------------------------------------------------------------
Fss = edfheader.frequency ; % Sampling frequencies
time_window = 10 ; % Time window decleration
Fs = Fss(1) ; % Sampling frequency
window_length = Fs * time_window ; % window length 
NumOfWindows = floor(length(EEGFpzCz)/window_length) ; % Number of windows
%------------------------------------------------------------------
t = 0:10:length (EEGFpzCz)/100-1 ;
%------------------------------------------------------------------
state = zeros(1,length(t)) ; % Declaring the states in each time window
    for i = 2 : length (hypnogram_time)-2
        state (hypnogram_time(i)/10:(hypnogram_time(i+1)/10-1)) = hypnogram_state (i) ;
    end
%------------------------------------------------------------------


X = zeros(NumOfWindows,10) ;

for i = 0:NumOfWindows-1
    X (i+1,1) = bandpower (EEGFpzCz(window_length*i+1:window_length*(i+1)),Fs,[0.5 4]) ; % Delta band power of EEGFpzCz
    X (i+1,2) = bandpower (EEGFpzCz(window_length*i+1:window_length*(i+1)),Fs,[4 7])  ; % Theta band power of EEGFpzCz
    X (i+1,3) = bandpower (EEGFpzCz(window_length*i+1:window_length*(i+1)),Fs,[8 15]) ; % Alpha band power of EEGFpzCz
    X (i+1,4) = bandpower (EEGFpzCz(window_length*i+1:window_length*(i+1)),Fs,[16 31]) ; % Beta band power of EEGFpzCz
    X (i+1,5) = bandpower (EEGPzOz(window_length*i+1:window_length*(i+1)),Fs,[0.5 4])  ; % Delta band power of EEGPzOz
    X (i+1,6) = bandpower (EEGPzOz(window_length*i+1:window_length*(i+1)),Fs,[4 7]) ; % Theta band power of EEGPzOz
    X (i+1,7) = bandpower (EEGPzOz(window_length*i+1:window_length*(i+1)),Fs,[8 15]) ; % Alpha band power of EEGPzOz
    X (i+1,8) = bandpower (EEGPzOz(window_length*i+1:window_length*(i+1)),Fs,[16 31]) ; % Beta band power of EEGPzOz
    X (i+1,9) = bandpower (EOGhorizontal(window_length*i+1:window_length*(i+1))) ; % power of EOGhorizontal
    X (i+1,10) = bandpower (EMGsubmental(window_length*i+1:window_length*(i+1))) ; % power of EMGsubmental
end
 end
 