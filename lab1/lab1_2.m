% Part 2(a): Aliasing demonstration
F1 = 300e6;  % 300MHz
F2 = 800e6;  % 800MHz
Fs = 500e6;  % 500MHz sampling rate

% Time vector for continuous signal
t = 0:1/(100*Fs):5/F1;
x1_t = cos(2*pi*F1*t);
x2_t = cos(2*pi*F2*t);

% Sampling times
t_sampled = 0:1/Fs:5/F1;
x1_sampled = cos(2*pi*F1*t_sampled);
x2_sampled = cos(2*pi*F2*t_sampled);

% Plot signals
figure;
subplot(2,1,1);
plot(t, x1_t, 'b-'); hold on;
plot(t_sampled, x1_sampled, 'r.', 'MarkerSize', 10);
title('Signal x1(t) and its samples');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
legend('Continuous', 'Sampled');

subplot(2,1,2);
plot(t, x2_t, 'b-'); hold on;
plot(t_sampled, x2_sampled, 'r.', 'MarkerSize', 10);
title('Signal x2(t) and its samples');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
legend('Continuous', 'Sampled');


exportgraphics(gcf, 'Sample1.png', 'Resolution', 300)
