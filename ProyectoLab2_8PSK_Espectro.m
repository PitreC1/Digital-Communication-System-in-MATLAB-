close all
clear all

%% FUENTE BINARIA
img=imread("panda3.jpg");

%Imagen original en escala de grises
img_Gray =  rgb2gray(img);
figure(1),imshow(img_Gray);
title('imagen original ');
colorbar;

%Imagen Binarizada
level=graythresh(img_Gray);
img_Binaria=im2bw(img_Gray,level);
figure(2),imshow(img_Binaria);

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


%% MODULACION EN BANDA BASE 8PSK

alfabeto = [0,0,0% S1
    0,0,1% S2
    0,1,0% S3
    0,1,1% S4
    1,0,0% S5
    1,0,1% S6
    1,1,0% S7
    1,1,1]; % S8

Simbolos = [-0.707 - 0.707j,
    -1 + 0j,
    0 + 1j,
    -0.707 + 0.707j,
    0 - 1j,
    0.707 - 0.707j,
    +0.707 + 0.707j,
    1 + 0j];

% Transformacion de la secuencia de bits a secuencia de simbolos de S1 a S8
grupo = 3;
numero_Simbolos = numel(secuencia) / grupo;
secuencia_Simbolos = zeros(1, numero_Simbolos);

for i = 1:numero_Simbolos
    indiceInicio = (i - 1) * grupo + 1;
    indiceFin = indiceInicio + grupo - 1;
    grupoBits = secuencia(indiceInicio:indiceFin);

    for j = 1:size(alfabeto, 1)
        if isequal(grupoBits, alfabeto(j, :))
            secuencia_Simbolos(i) = j;
            break;
        end
    end
end

%Transformacion a secuencia de simbolos rectangulares

secuencia_simbolos_rect= zeros(1,numero_Simbolos);
for i = 1:numero_Simbolos
    posicion = secuencia_Simbolos(i);
    secuencia_simbolos_rect(i) = Simbolos(posicion);
end
% Grafica de constelacion usando la secuencia de simbolos rectangulares
scatterplot(secuencia_simbolos_rect)
title('Diagrama de Constelación 8PSK secuencia');
xlabel('Parte Real');
ylabel('Parte Imaginaria');
axis square;
grid on;


%% Espectro Señal Modulada en Banda Base

fs = (5*10^4)/3;
[XmBB,f] = FourierT(secuencia_simbolos_rect,fs);
figure(5)
plot(f, XmBB);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal');

%% PULSO CONFORMADOR

%% Funcion para determinar el Espectro

function [X,f] = FourierT(x,fs)

    %t = (0:(length(x)-1)) / fs;
    senal = 2 * x - 1;
    X = fft(senal);
    X = fftshift(X);
    X = abs(X);
    f = (-length(x)/2:length(x)/2-1) * fs / length(x);
end


