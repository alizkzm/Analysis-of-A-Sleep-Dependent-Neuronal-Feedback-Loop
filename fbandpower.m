function output =  fbandpower (signal,LowerBand,UpperBand,Fs)
T = 1/Fs;
L = length(signal) ;
F = fft(signal) ;
P2 = abs (F/L) ;
P1 = P2(1:L/2+1);
power = P1.^2 ;
power = power ((LowerBand*Fs/L+1):(UpperBand*Fs/L)) ;
power(2:end-1) = 2*power(2:end-1);
output = sum(power) ;
end