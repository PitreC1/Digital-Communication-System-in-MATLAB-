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


%% MODULACION EN BANDA BASE PSK
% Parámetros de la modulación PSK
M=8;% Numero de simbolos 8PSK 
fp=100;% frecuencia potadora 
fs=10*fp;% frecuencia de muestreo 

% Generar señal portadora
t = 0:1/fs:length(secuencia)/fs - 1/fs;
portadora = cos(2*pi*fp*t);

% Convertir bits en símbolos
simbolos = bi2de(reshape(secuencia, log2(M), []).', 'left-msb');

% % Modulación por desplazamiento de fase (PSK)
% fase = 2*pi*(simbolos/M);
% modulada = cos(2*pi*fp*t + fase);

% Visualización de las señales
figure(4),
plot(t, secuencia);
xlabel('Tiempo SEÑAL BITS');
ylabel('Valor');

figure(5),
plot(t, modulada, 'b');
xlabel('Tiempo');
ylabel('Amplitud modulada');

% Configuración de los ejes para una mejor visualización
axis([0 t(end) -1.5 1.5]);

% Mostrar constelación de símbolos
figure(6), plot(cos(fase), sin(fase), 'o');
xlabel('Parte Real CONSTELACION');
ylabel('Parte Imaginaria');
axis square;
grid on;

%% PILSO CONFORMADOR 
