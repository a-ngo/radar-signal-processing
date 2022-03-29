%% Range Estimation
% TODO : Find the Bsweep of chirp for 1 m resolution


% TODO : Calculate the chirp time based on the Radar's Max Range


% TODO : define the frequency shifts 



calculated_range = 100;

% Display the calculated range
% disp("calculated_range: " + calculated_range);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Doppler Velocity Calculation
c = 3*10^8;         %speed of light
frequency = 77e9;   %frequency in Hz

% TODO: Calculate the wavelength
lambda = c / frequency;


% TODO: Define the doppler shifts in Hz using the information from above 
fd = [3e3 -4.5e3 11e3 -3e3];


% TODO: Calculate the velocity of the targets  fd = 2*vr/lambda
vr = fd / 2 * lambda;


% TODO: Display results
% disp("calculated velocites: " + vr);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1D FFT - range fft

% f_sample = 1e3;         % in Hz
% signal_duration = 1.3;  % in sec
% 
% 1. define a signal
% signal = A*sin(2*pi*f*t);
% 
% 2. run FFT for the dimension of samples N
% signal_fft = fft(signal, N);
% 
% 3. output of FFT is a complex number a+jb. take here the absolute value
% bc here only care for the magnitude -> sqrt(a^2 + b^2)
% signal_fft = abs(signal_fft);
% 
% 4. FFT output generates a mirror image of the signal. but we are only
% interested in the positive half of signal length L since it is the
% replica of the negative half and has all the information we need
% signal_fft = signal_fft(1:L/2+1);
% 
% 5. plot this output against frequency

Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector

% TODO: Form a signal containing a 77 Hz sinusoid of amplitude 0.7 and a 43Hz sinusoid of amplitude 2.
S = 0.7*sin(2*pi*77*t) + 2*sin(2*pi*43*t);

% Corrupt the signal with noise 
X = S + 2*randn(size(t));

% Plot the noisy signal in the time domain. It is difficult to identify the frequency components by looking at the signal X(t). 
% figure
% plot(1000*t(1:50) ,X(1:50))
% title('Signal Corrupted with Zero-Mean Random Noise')
% xlabel('t (milliseconds)')
% ylabel('X(t)')

% TODO : Compute the Fourier transform of the signal. 
signal_fft = fft(S);

% TODO : Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
P2 = abs(signal_fft/L);
P1  = P2(1:L/2+1);

% Plotting
f = Fs*(0:(L/2))/L;
% figure
% plot(f,P1) 
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% 2D FFT - Range Doppler Map

%% Part 1 : 1D FFT

% Generate Noisy Signal

% Specify the parameters of a signal with a sampling frequency of 1 kHz 
% and a signal duration of 1.5 seconds.

Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector

% Form a signal containing a 50 Hz sinusoid of amplitude 0.7 and a 120 Hz 
% sinusoid of amplitude 1.

S = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);

% Corrupt the signal with zero-mean white noise with a variance of 4
X = S + 2*randn(size(t));

% Plot the noisy signal in the time domain. It is difficult to identify
% the frequency components by looking at the signal X(t).

figure(1);
tiledlayout(1,2)

% left plot
nexttile
plot(1000*t(1:50), X(1:50))
title('Signal corrupted with Zero-Mean Random Noise')
xlabel('t (milliseconds)')
ylabel('X(t)')

% Compute the Fourier Transform of the Signal.

Y = fft(X);

% Compute the 2 sided spectrum P2. Then compute the single-sided spectrum
% P1 based on P2 and the even-valued signal length L.

P2 = abs(Y/L);
P1 = P2(1:L/2+1);

% Define the frequency domain f and plot the single-sided amplitude 
% spectrum P1. The amplitudes are not exactly at 0.7 and 1, as expected,
% because of the added noise. On average, longer signals produce better 
% frequency approximations

f = Fs*(0:(L/2))/L;

nexttile
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

% saveas(gcf, 'fft_1d.png')

%% Part 2 - 2D FFT

% Implement a second FFT along the second dimension to determine the 
% Doppler frequency shift.

% First we need to generate a 2D signal
% Convert 1D signal X to 2D using reshape

% while reshaping a 1D signal to 2D we need to ensure that dimensions match
% length(X) = M*N

% let
M = length(X)/50;
N = length(X)/30;

X_2d = reshape(X, [M, N]);

figure(2);
tiledlayout(1,2)

nexttile
imagesc(X_2d)

% Compute the 2-D Fourier transform of the data. Shift the zero-frequency 
% component to the center of the output, and plot the resulting 
% matrix, which is the same size as X_2d.

Y_2d = fft2(X_2d);

nexttile
imagesc(abs(fftshift(Y)))

% saveas(gcf, 'fft_2d.png')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% CFAR

% Implement 1D CFAR using lagging cells on the given noise and target scenario.

% Close and delete all currently open figures
close all;

% Generate Noisy Signal

% Specify the parameters of a signal with a sampling frequency of 1 kHz 
% and a signal duration of 1.5 seconds.

Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector

% Form a signal containing a 50 Hz sinusoid of amplitude 0.7 and a 120 Hz 
% sinusoid of amplitude 1.

S = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);

% Corrupt the signal with zero-mean white noise with a variance of 4
X = S + 2*randn(size(t));

X_cfar = abs(X);

% Data_points
Ns = 1500;  % let it be the same as the length of the signal

%Targets location. Assigning bin 100, 200, 300, and 700 as Targets
%  with the amplitudes of 16, 18, 27, 22.
X_cfar([100 ,200, 300, 700])=[16 18 27 22];

% plot the output
figure(1);
tiledlayout(2,1)
nexttile
plot(X_cfar)

% Apply CFAR to detect the targets by filtering the noise.

% TODO: Define the number of Training Cells
T = 12;
% TODO: Define the number of Guard Cells 
G = 4;
% TODO: Define Offset (Adding room above noise threshold for the desired SNR)
offset = 5;

% Initialize vector to hold threshold values 
threshold_cfar = zeros(Ns-(G+T+1),1);

% Initialize Vector to hold final signal after thresholding
signal_cfar = zeros(Ns-(G+T+1),1);

% Slide window across the signal length
for i = 1:(Ns-(G+T+1))     

    % TODO: Determine the noise threshold by measuring it within
    % the training cells
    noise_level = sum(X_cfar(i:i+T-1));
    % TODO: scale the noise_level by appropriate offset value and take
    % average over T training cells
    threshold = (noise_level/T)*offset;
    % Add threshold value to the threshold_cfar vector
    threshold_cfar(i) = threshold;
    % TODO: Measure the signal within the CUT
    signal = X_cfar(i+T+G);
    % add signal value to the signal_cfar vector
    signal_cfar(i) = signal;
end

% plot the filtered signal
plot(signal_cfar);
legend('Signal')

% plot original sig, threshold and filtered signal within the same figure.
nexttile
plot(X_cfar);
hold on
plot(circshift(threshold_cfar,G),'r--','LineWidth',2)
hold on
plot (circshift(signal_cfar,(T+G)),'g--','LineWidth',2);
legend('Signal','CFAR Threshold','detection')

% saveas(gcf, 'cfar.png')





