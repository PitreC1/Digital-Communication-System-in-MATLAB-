clear all 
close all
%% FUENTE BINARIA
img = imread("panda3.jpg");

% Imagen original en escala de grises
img_Gray = rgb2gray(img);

% Imagen Binarizada
level = graythresh(img_Gray);
img_Binaria = im2bw(img_Gray, level);

% Secuencia de bits de la imagen
secuencia = reshape(img_Binaria, 1, []);

%Graficas Fuente
figure(1),
imshow(img_Gray);
title('Imagen en escala de grises ');
colorbar;

figure(2),
imshow(img_Binaria);
title('Imagen Binarizada ');

%% MODULACION EN BANDA BASE 16 QAM
longitud_secuencia = length(secuencia);
longitud_multiplo_4 = ceil(longitud_secuencia / 4) * 4;
secuencia = [secuencia, zeros(1, longitud_multiplo_4 - longitud_secuencia)];

% Calcula la cantidad de símbolos
cantidad_simbolos = longitud_multiplo_4 / 4;
Ns=length(secuencia);
c=8; %cantidad de ceros que toman los filtros p(t) y q(t)

% Modulación 16-QAM
M = 16;
Mpam = sqrt(M);
d = 1;
A = zeros(1, Mpam);
for i = 1:Mpam
    A(i) = (2 * i - 1 - Mpam) * d;
end

% Calculo de la energía de símbolo
P = perms([A A]);
p2 = reshape(P', [2, numel(P) / 2])';
p3 = unique(p2, 'rows');
dr = p3(:, 1)';
di = p3(:, 2)';
Es = sum(((dr.^2) + (di.^2)) ./ M);

% Vectores de símbolos en paralelo que entran al filtro conformador
S1 = randsrc(1, cantidad_simbolos, A);
S2 = randsrc(1, cantidad_simbolos, A);
simbolos = S1 + 1j * S2;


%% PULSO CONFORMADOR
% Parámetros del pulso conformador
alfa = 0.4; %factor de roll-off
span = c; %numero de simbolos
mps = 7; %muestras por simbolos

S1=[S1 zeros(1,2*c)];
S2=[S2 zeros(1,2*c)];

% Pulso conformador 
pulso = rcosdesign(alfa, span, mps,'sqrt');

%sobremuestreo
X1 = upsample(S1, mps);
X2= upsample(S2, mps);

% conformar pulsos
pulsos_conf_real = filter(pulso,1,X1);
pulsos_conf_img = filter(pulso,1,X2);


% Grafica Pulso conformador
figure(6),
subplot(3,1,1)
stem(pulso);
title('Pulsos Conformador');
xlabel('Muestras');
ylabel('Amplitud');

subplot(3,1,2)
stem(X1)
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
stem(X2)
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


%% Modulacion Pasa Banda 
Rs=1;
fs=mps*Rs;
ts=1/fs;
Tb=1;
fc=2*Rs;

t = 0:ts:(length(pulsos_conf_real)*Tb - 1/fc)/fs;

%Señal Pasa Banda 
senal_real=sqrt(2)*pulsos_conf_real.*cos(2*pi*fc.*t);
senal_img=-sqrt(2)*pulsos_conf_img.*sin(2*pi*fc.*t);
senal_tx=senal_real + senal_img; 

%Grafica Modulacion Pasa Banda
% figure(9),
% plot(t, senal_tx);
% title('Señal a Transmitir');
% xlabel('Tiempo (s)');
% xlim([0,0.1]);
% ylabel('Amplitud');
% grid on; 

% ESPECTRO DE LA SEÑAL MODULADA EN PASA BANDA 

[XmBB,f] = FourierT(senal_tx,fs);
figure(10)
plot(f, XmBB);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal Modulada en Pasa Banda');

% Ruido del canal 
ebno=100;
sigma=sqrt(Es/(2*log2(M)*ebno));%determina la varianza de ruido
Z=sigma*randn(1,length(senal_tx));
Y=senal_tx+Z;
figure(2)
subplot(211),plot(senal_tx,'k'),grid on, title('Señal Modulada'),xlim([mps*50 mps*100])
subplot(212),plot(Y,'g'),grid on, title('Señal a la salida del canal'),xlim([mps*50 mps*100])

%DIAGRAMA DE CONSTELACION
Y_Real = senal_real+Z*300;
Y_Img = (senal_img+Z*300);
Y_simbolos = Y_Real - j.* Y_Img;
scatterplot(Y_simbolos);
title('Diagrama de Constelación Despues del Canal AWGN');
xlabel('Parte Real');
ylabel('Parte Imaginaria');
axis square;
grid on;

%% MUTITRAYECTORIA

alpha1 = 0.1; % Constante de escala
t1 = 10; % Desplazamiento en el tiempo (segundos)

Y_Multitrayecto_1 = alpha1*interp1(t, Y, t - t1, 'linear', 0); % Interpolación lineal

figure(13);
subplot(2,1,1);
plot(Y);
xlim([mps*50 mps*300]);
title('Señal Original x(t)');
xlim([0,800]);
xlabel('Tiempo (s)');
ylabel('Amplitud');

subplot(2,1,2);
plot(Y_Multitrayecto_1,'r');
xlim([mps*50 mps*300]);
title('Señal Trayecto 1');
xlabel('Tiempo (s)');
ylabel('Amplitud');
xlim([0,800]);

Y_Canal = Y + Y_Multitrayecto_1;

figure(14)
plot(Y_Canal)
xlim([mps*50 mps*200])
title('Señal a la Salida del Canal');

% ESPECTRO DE LA SEÑAL A LA SALIDA DEL CANAL 
[YAWGNM,f] = FourierT(Y_Canal,fs);
figure(16)
plot(f, YAWGNM);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal a la Salida del Canal');

%% Ecualizador

Y_Ecualizado1 = (Y_Canal) - (alpha1*interp1(t, Y_Canal, t - t1, 'linear', 0)+(alpha1^2)*interp1(t, Y_Canal, t - 2*t1, 'linear', 0));
Y_Ecualizado2 = Y_Ecualizado1 + (alpha1^2)*interp1(t, Y_Canal, t - 2*t1, 'linear', 0)+(alpha1^3)*interp1(t, Y_Canal, t - 3*t1, 'linear', 0);

figure(17)
subplot(2,1,1);
plot(Y)
xlim([mps*50 mps*100])
title('Señal con AWGN');
subplot(2,1,2);
plot(Y_Ecualizado2)
xlim([mps*50 mps*100])
title('Señal Ecualizada');
%% Demodulacion Pasa Banda
senal_Y1 = sqrt(2)*Y_Ecualizado2.*cos(2*pi*fc.*t);
senal_Y2 = -sqrt(2)*Y_Ecualizado2.*sin(2*pi*fc.*t);
senal_YDM = senal_Y1 + senal_Y2;

% ESPECTRO DE LA SEÑAL DEMODULADA 
[YDM,f] = FourierT(senal_YDM,fs);
figure(18)
plot(f, YDM)
xlabel('Frecuencia (Hz)');
xlim([-10,10]);
ylabel('Amplitud');
title('Espectro de la señal Demodulada');

%% Filtro Acoplado
filtro_Y1 = filter(pulso,1,senal_Y1);
filtro_Y2 = filter(pulso,1,senal_Y2);
figure(19)
plot(filtro_Y1,filtro_Y2)

%% Demodulacion Banda Base
Y1 = downsample(filtro_Y1,mps);
Y2 = downsample(filtro_Y2,mps);

eyediagram([filtro_Y1(5000:55000)' filtro_Y2(5000:55000)'],2*fs)
title('Diagrama del ojo señal rx')
xlim([0,0.1]);
scatterplot(Y1+1j*Y2),title('simbolos recibidos')


%% De-mapeo
% DE-MAPEO - Criterio de distancia mínima
T = Mpam - 1; % Número de umbrales de decisión
Tes = zeros(1, T);
for i = 1:T
    Tes(1, i) = (A(1, i) + A(1, i + 1)) / 2;
end
Se1 = Y1;
Se2 = Y2;
% De-mapeo
for i = 1:T
    Se1(Se1 < Tes(1, i)) = A(1, i);
    Se2(Se2 < Tes(1, i)) = A(1, i);
end
Se1(Se1 > Tes(1, T)) = A(1, Mpam);
Se2(Se2 > Tes(1, T)) = A(1, Mpam);

% Ajuste de longitud de señales
Se1 = Se1(2 * c + 1:length(Se1));
Se2 = Se2(2 * c + 1:length(Se2));
S1 = S1(1:length(S1) - 2 * c);
S2 = S2(1:length(S2) - 2 * c);

% SER
[Nerr, SER] = symerr(S1 + 1j * S2, Se1 + 1j * Se2);

%% Recuperacion de la señal 
% Combina S1 y S2 en una señal compleja
received_signal = S1 + 1j * S2;

secuencia_bits = mapeo_Simbolo(received_signal);
secuencia_bits=double(secuencia_bits)-48;
alto = 200;
ancho = numel(secuencia_bits) / alto;
ancho = floor(ancho);
alto = numel(secuencia_bits) / ancho;
matriz_bits = reshape(secuencia_bits, alto, ancho);
valores_pixeles = matriz_bits * 255;  %1 representa el blanco y 0 representa el negro
imagen_escala_grises = uint8(valores_pixeles);  % Convertir a tipo de datos uint8 para crear la imagen

% Mostrar la imagen
figure(21)
imshow(imagen_escala_grises);
title('Imagen generada a partir de la secuencia de bits');

%% Funcion para determinar el Espectro

function grupo_Bits = mapeo_Simbolo(valor_complejo)
    
    % Define un diccionario (matriz de celdas) para mapear valores a grupos de bits
    diccionario = {
        -3 + 3i, '0000';
        -1 + 3i, '0001';
        1 + 3i, '0011';
        3 + 3i, '0010';
        -3 + 1i, '0100';
        -1 + 1i, '0101';
        1 + 1i, '0111';
        3 + 1i, '0110';
        -3 - 1i, '1110';
        -1 - 1i, '1111';
        1 - 1i, '1101';
        3 - 1i, '1100';
        -3 - 3i, '1000';
        -1 - 3i, '1001';
        1 - 3i, '1011';
        3 - 3i, '1010'
    };

    % Busca el grupo de bits correspondiente en el diccionario
    grupo_Bits = '';

    for i = 1:length(valor_complejo)
        index = find(cellfun(@(x) isequal(x, valor_complejo(i)), diccionario(:,1)));
        if ~isempty(index)
            grupo_Bits = [grupo_Bits, diccionario{index, 2}];
        else
            % En caso de que el valor complejo no exista en el diccionario, puedes asignar un valor predeterminado o manejar el error según tus necesidades
            error('Valor complejo no válido');
        end
    end
end


function [X,f] = FourierT(x,fs)

    X = fft(x);
    X = fftshift(X);
    X = abs(X);
    f = (-length(x)/2:length(x)/2-1) * fs / length(x);
end
