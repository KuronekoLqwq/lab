clear;
close all;
clc;

% 1.a 
% Example FIR filter (Moving average)
b_fir = ones(1,5);  % 5-point moving average
a_fir = 1;

% Plot results
figure;
% Get frequency response
[h_fir,w_fir] = freqz(b_fir, a_fir);

% Magnitude response
subplot(2,2,1);
plot(w_fir/pi, 20*log10(abs(h_fir)));
grid on;
xlabel('Normalized Frequency (\times\pi rad/sample)');
ylabel('Magnitude (dB)');
title('FIR - Magnitude Response');

% Phase response
subplot(2,2,3);
plot(w_fir/pi, angle(h_fir));
grid on;
xlabel('Normalized Frequency (\times\pi rad/sample)');
ylabel('Phase (radians)');
title('FIR - Phase Response');

% Pole-zero plot
subplot(2,2,[2,4]);  % Make pole-zero plot occupy right half
zplane(b_fir, a_fir);
title('FIR - Pole-Zero Plot');

% Overall title
sgtitle('FIR Responce', 'FontSize', 12, 'FontWeight', 'bold');

exportgraphics(gcf, 'FIR.png', 'Resolution', 300)
% Example IIR filter (First-order lowpass)
b_iir = [1 1];
a_iir = [1 -1];

figure;
% Get frequency response
[h_iir,w_iir] = freqz(b_iir, a_iir);

% Magnitude response
subplot(2,2,1);
plot(w_iir/pi, 20*log10(abs(h_iir)));
grid on;
xlabel('Normalized Frequency (\times\pi rad/sample)');
ylabel('Magnitude (dB)');
title('IIR - Magnitude Response');

% Phase response
subplot(2,2,3);
plot(w_iir/pi, angle(h_iir));
grid on;
xlabel('Normalized Frequency (\times\pi rad/sample)');
ylabel('Phase (radians)');
title('IIR - Phase Response');

% Pole-zero plot
subplot(2,2,[2,4]);  % Make pole-zero plot occupy right half
zplane(b_iir, a_iir);
title('Pole-Zero Plot');

% Overall title
sgtitle('IIR Responce', 'FontSize', 12, 'FontWeight', 'bold');

exportgraphics(gcf, 'IIR.png', 'Resolution', 300)% 1.b
