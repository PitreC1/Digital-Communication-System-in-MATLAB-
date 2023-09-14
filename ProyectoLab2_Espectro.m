close all
clear all

%% FUENTE BINARIA
img=imread("panda3.jpg");

%Imagen original en escala de grises
img_Gray =  rgb2gray(img);
%figure(1),imshow(img_Gray);
title('imagen original ');
colorbar;

%Imagen Binarizada
level=graythresh(img_Gray);
img_Binaria=im2bw(img_Gray,level);
%figure(2),imshow(img_Binaria);

%Secuencia de bits de la imagen
secuencia = reshape(img_Binaria, 1, []);

%Grafica señal a transmitir en el Dominio del Tiempo
figure(3)
title('señal en el dominio el tiempo');
plot(secuencia,'linewidth',2),grid on;

%% Espectro señal a transmitir

fs = 2*5*10^4;
[X,f] = FourierT(secuencia,fs);
plot(f, X);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal');

%% Funcion para determinar el Espectro

function [X,f] = FourierT(x,fs)

    %t = (0:(length(x)-1)) / fs;
    senal = 2 * x - 1;
    X = fft(senal);
    X = fftshift(X);
    X = abs(X);
    f = (-length(x)/2:length(x)/2-1) * fs / length(x);
end



