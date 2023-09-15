
close all
clear all


%% FUENTE BINARIA
img=imread("panda3.jpg");

%Imagen original en escala de grises
img_Gray =  rgb2gray(img);

%Imagen Binarizada
level=graythresh(img_Gray);
img_Binaria=im2bw(img_Gray,level);

%Secuencia de bits de la imagen
secuencia = reshape(img_Binaria, 1, []);

%Graficas Fuente
figure(1),
imshow(img_Gray);
title('Imagen en escala de grises ');
colorbar;

figure(2),
imshow(img_Binaria);
title('Imagen Binarizada ');

%% %% ESPECTRO DE LA SEÑAL A TRANSMITIR

fs = 2*5*10^4;
senial = 2 * secuencia - 1;
[X,f] = FourierT(secuencia,fs);
figure(3)
plot(f, X);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal a transmitir');

%% MODULACION EN BANDA BASE 8PSK

% Transformacion de la secuencia de bits a secuencia de simbolos de S1 a S8
grupo = 3;
numero_Simbolos = numel(secuencia) / grupo;
secuencia_Simbolos = zeros(1, numero_Simbolos); 

for i = 1:numero_Simbolos
    indiceInicio = (i - 1) * grupo + 1;
    indiceFin = indiceInicio + grupo - 1;
    grupoBits = secuencia(indiceInicio:indiceFin);

    secuencia_Simbolos(i) = mapeo_Simbolos(grupoBits);
end

%Transformacion a secuencia de simbolos rectangulares

secuencia_simbolos_real= real(secuencia_Simbolos);
secuecia_simbolos_img= imag(secuencia_Simbolos);

% Grafica de constelacion usando la secuencia de simbolos rectangulares
scatterplot(secuencia_Simbolos);
title('Diagrama de Constelación 8PSK');
xlabel('Parte Real');
ylabel('Parte Imaginaria');
axis square;
grid on;

%% ESPECTRO DE LA SEÑAL A TRANSMITIR EN BANDA BASE 

fs = 2*5*10^4;
%senial = 2 * secuencia - 1;
[X,f] = FourierT(secuencia_Simbolos,fs);
figure(5)
plot(f, X);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal Modulada en Banda Base');

%% PULSO CONFORMADOR
% Parámetros del pulso conformador
alfa = 0.5; %factor de roll-off
span = 1; %numero de simbolos
mps = 6; %muestras por simbolos

% Pulso conformador 
pulso = rcosdesign(alfa, span, mps, 'sqrt');

%sobremuestreo
secuencia_Sobremuestreada_r = upsample(secuencia_simbolos_real, mps+1);
secuencia_Sobremuestreada_i = upsample(secuecia_simbolos_img, mps+1);
secuencia_Sobremuestreada = upsample(secuencia_Simbolos, mps);% quitar +1

% conformar pulsos
pulsos_conf_real = conv(secuencia_Sobremuestreada_r, pulso);
pulsos_conf_img = conv(secuencia_Sobremuestreada_i, pulso);
pulsos_conformados = conv(secuencia_Sobremuestreada, pulso);

% Grafica Pulso conformador
figure(6),
subplot(3,1,1)
stem(pulso);
title('Pulsos Conformador');
xlabel('Muestras');
ylabel('Amplitud');

subplot(3,1,2)
stem(secuencia_Sobremuestreada_r)
title('Secuencia sobremuestreada real');
xlabel('Muestras');
xlim([0,33]);
ylabel('Amplitud');

subplot(3,1,3)
stem(pulsos_conf_real);
title('Señal Pulsos conformados reales');
xlabel('Muestras');
xlim([0,33]);
ylabel('Amplitud');

% Graficas
figure(7),
subplot(3,1,1)
stem(pulso);
title('Pulsos Conformador');
xlabel('Muestras');
ylabel('Amplitud');

subplot(3,1,2)
stem(secuencia_Sobremuestreada_i)
title('Secuencia sobremuestreada imaginaria');
xlabel('Muestras');
xlim([0,33]);
ylabel('Amplitud');

subplot(3,1,3)
stem(pulsos_conf_img);
title('Señal Pulsos conformados imaginarios');
xlabel('Muestras');
xlim([0,33]);
ylabel('Amplitud');

%% ESPECTRO DE LA SEÑAL A LA SALIDA DEL PULSO CONFORMADOR

fs = (2*10^4)/3;
[XmBB,f] = FourierT(pulsos_conformados,fs);
figure(8)
plot(f, XmBB);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal Modulada en Banda Base');

%% Modulacion Pasa Banda 
Rs=10^6;
fs=mps*Rs;
ts=1/fs;
Tb=1;
fc=2*Rs;

t = 0:ts:(length(pulsos_conf_real)*Tb - 1/fc)/fs;

% Moduladora
moduladora_PB=pulsos_conf_real;

%Señal Pasa Banda 
senal_real=sqrt(2)*pulsos_conf_real.*cos(2*pi*fc.*t);
senal_img=sqrt(2)*pulsos_conf_img.*sin(2*pi*fc.*t);
senal_tx=senal_real - senal_img; %Señal a transmitir


%Grafica Modulacion Pasa Banda
figure(9),
plot(t, senal_tx);
title('Señal a Transmitir');
xlabel('Tiempo (s)');
xlim([0,.016]);
ylabel('Amplitud');
grid on; 


%% ESPECTRO DE LA SEÑAL MODULADA EN PASA BANDA FSK

% fs = (5*10^4)/3;
[XmBB,f] = FourierT(senal_tx,fs);
figure(10)
plot(f, XmBB);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal Modulada en Pasa Bana');

%% Funciones, etc

function mapeo = mapeo_Simbolos(grupo_Bits)
    
    % Define un diccionario (matriz de celdas) para mapear los grupos de bits a valores
    diccionario_8PSK = {
    '000', 1 + 0j;     % S1
    '001', 0.707 + 0.707j;  % S2
    '010', 0 + 1j;     % S3
    '011', -0.707 + 0.707j;  % S4
    '100', -1 + 0j;     % S5
    '101', -0.707 - 0.707j; % S6
    '110', 0 - 1j;     % S7
    '111', 0.707 - 0.707j; % S8
    };


    % Convierte el grupo_Bits a una cadena
    grupo_Bits_str = sprintf('%d', grupo_Bits);

    % Busca el valor correspondiente en el diccionario
    index = find(strcmp(diccionario_8PSK(:,1), grupo_Bits_str));
    
    if ~isempty(index)
        mapeo = diccionario_8PSK{index, 2};
    else
        error('Grupo de bits no válido');
    end
end

% Funcion para determinar el Espectro

function [X,f] = FourierT(x,fs)

    X = fft(x);
    X = fftshift(X);
    X = abs(X);
    f = (-length(x)/2:length(x)/2-1) * fs / length(x);
end








