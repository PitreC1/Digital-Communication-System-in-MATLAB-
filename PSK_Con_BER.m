
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

% ESPECTRO DE LA SEÑAL A TRANSMITIR

fs = 2*5*10^4;
senial = 2 * secuencia - 1;
[X,f] = FourierT(secuencia,fs);
figure(3)
plot(f, X);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
xlim([-20000,20000]);
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

    [secuencia_Simbolos(i),Energia] = mapeo_Simbolos(grupoBits);

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

% ESPECTRO DE LA SEÑAL A TRANSMITIR EN BANDA BASE

fs = 2*5*10^4;
%senial = 2 * secuencia - 1;
[X,f] = FourierT(secuencia_Simbolos,fs);
figure(5)
plot(f, X);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
xlim([-20000,20000]);
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
secuencia_Sobremuestreada = upsample(secuencia_Simbolos, mps+1);

% conformar pulsos
pulsos_conf_real = filter(pulso,1,secuencia_Sobremuestreada_r);
pulsos_conf_img = filter(pulso,1,secuencia_Sobremuestreada_i);
pulsos_conformados = filter(pulso,1,secuencia_Sobremuestreada);

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


% ESPECTRO DE LA SEÑAL A LA SALIDA DEL PULSO CONFORMADOR

fs = (2*10^4)/3;
[XmBB,f] = FourierT(pulsos_conformados,fs);
figure(8)
plot(f, XmBB);
xlabel('Frecuencia (Hz)');
xlim([-2000,2000]);
ylabel('Amplitud');
title('Espectro de la señal Modulada en Banda Base');

%% Modulacion Pasa Banda
Rs=10;
fs=mps*Rs;
ts=1/fs;
Tb=1;
fc=2*10^6;

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
xlim([0,10]);
ylabel('Amplitud');
grid on;

% ESPECTRO DE LA SEÑAL MODULADA EN PASA BANDA

% fs = (5*10^4)/3;
[XmBB,f] = FourierT(senal_tx,fs);
figure(10)
plot(f, XmBB);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal Modulada en Pasa Banda');

%% CANAL AWGN

M=8; %Orden de Modulación
EbNoVec = (5:15);

% Inicializa un vector para almacenar las BER estimadas en cada iteración
BER = zeros(size(EbNoVec));

% Inicializa vectores para bits transmitidos y recibidos

bits_transmitidos_total = [];
bits_recibidos_total = [];


%% Iteración 

for i = 1:length(EbNoVec)

    ebno = 10.^((EbNoVec(i))/(10));
    sigma=sqrt(Energia/(2*log2(M)*ebno));%determina la varianza de ruido
    Z=sigma*randn(1,length(senal_tx));
    Y=senal_tx;
    
    if i ==length(EbNoVec)
        figure(11),
        subplot(211),plot(senal_tx),grid on, title('Señal Modulada'),xlim([mps*50 mps*200])
        subplot(212),plot(Y,'b'),grid on, title('Señal a través de canal AWGN'),xlim([mps*50 mps*200])
    
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

    end
    % % MUTITRAYECTORIA
    % 
    % alpha1 = 0.1; % Constante de escala
    % t1 = 1; % Desplazamiento en el tiempo (segundos)
    % 
    % Y_Multitrayecto_1 = alpha1*interp1(t, Y, t - t1, 'linear', 0); % Interpolación lineal
    % 
    % if i == length(EbNoVec)
    %     figure(13);
    %     subplot(2,1,1);
    %     plot(Y);
    %     xlim([mps*50 mps*300]);
    %     title('Señal Original x(t)');
    %     xlim([0,800]);
    %     xlabel('Tiempo (s)');
    %     ylabel('Amplitud');
    % 
    %     subplot(2,1,2);
    %     plot(Y_Multitrayecto_1,'r');
    %     xlim([mps*50 mps*300]);
    %     title('Señal Trayecto 1');
    %     xlabel('Tiempo (s)');
    %     ylabel('Amplitud');
    %     xlim([0,800]);
    % end
    % 
    % Y_Canal = Y + Y_Multitrayecto_1;
    % 
    % if i == length(EbNoVec)
    %     figure(14)
    %     plot(Y_Canal)
    %     xlim([mps*50 mps*200])
    %     title('Señal a la Salida del Canal');
    % 
    %     % ESPECTRO DE LA SEÑAL A LA SALIDA DEL CANAL
    %     [YAWGNM,f] = FourierT(Y_Canal,fs);
    %     figure(16)
    %     plot(f, YAWGNM);
    %     xlabel('Frecuencia (Hz)');
    %     ylabel('Amplitud');
    %     title('Espectro de la señal a la Salida del Canal');
    % 
    % end
    % % Ecualizador
    % 
    % Y_Ecualizado1 = (Y_Canal) - (alpha1*interp1(t, Y_Canal, t - t1, 'linear', 0)+(alpha1^2)*interp1(t, Y_Canal, t - 2*t1, 'linear', 0));
    % Y_Ecualizado2 = Y_Ecualizado1 + (alpha1^2)*interp1(t, Y_Canal, t - 2*t1, 'linear', 0)+(alpha1^3)*interp1(t, Y_Canal, t - 3*t1, 'linear', 0);
    % 
    % if i == length(EbNoVec)
    %     figure(17)
    %     subplot(2,1,1);
    %     plot(Y)
    %     xlim([mps*50 mps*100])
    %     title('Señal con AWGN');
    %     subplot(2,1,2);
    %     plot(Y_Ecualizado2)
    %     xlim([mps*50 mps*100])
    %     title('Señal Ecualizada');
    % end

    % Demodulacion Pasa Banda
    senal_Y1 = sqrt(2)*Y.*cos(2*pi*fc.*t);
    senal_Y2 = -sqrt(2)*Y.*sin(2*pi*fc.*t);
    senal_YDM = senal_Y1 + senal_Y2;
    
    if i == length(EbNoVec)
        % ESPECTRO DE LA SEÑAL DEMODULADA
        [YDM,f] = FourierT(senal_YDM,fs);
        figure(18)
        plot(f, YDM);
        xlabel('Frecuencia (Hz)');
        ylabel('Amplitud');
        title('Espectro de la señal Demodulada');
    end
    % Filtro Acoplado
    filtro_Y1 = filter(pulso,1,senal_Y1);
    filtro_Y2 = filter(pulso,1,senal_Y2);

    if i == length(EbNoVec)
        figure(19)
        plot(filtro_Y1,filtro_Y2)
    end
    % Demodulacion Banda Base
    Y1 = downsample(filtro_Y1,mps +1);
    Y2 = downsample(filtro_Y2,mps +1);

    if i == length(EbNoVec)
        eyediagram([filtro_Y1(5000:55000)' filtro_Y2(5000:55000)'],2*fs)
        title('Diagrama del ojo señal rx')
        xlim([0,0.1]);
        scatterplot(Y1+1j*Y2),title('simbolos recibidos')
    end

    % Demapeo
    num_simbolos_rx = numel(Y1);% Numero de sumbolor recibidos
    simbolos_binarios = {'000', '001', '010', '011', '100', '101', '110', '111'};
    bits_demapeados = cell(1, num_simbolos_rx);% Vector para bits demapeados

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

    % Para cada símbolo recibido se obtiene el simbolo complejo
    for n = 1:num_simbolos_rx
        simbolo_rx = Y1(n) + 1j * Y2(n);
        distancia_minima = inf; % distancia minima
        grupo_bits_recuperado = [];

        % Para cada símbolo de la constelación
        for j = 1:M
            distancia = abs(simbolo_rx - diccionario_8PSK{j, 2});% Calculo de la distancia para cada simbolo
            if distancia < distancia_minima
                distancia_minima = distancia;
                grupo_bits_recuperado = simbolos_binarios{j};
            end
        end
        bits_demapeados{n} = grupo_bits_recuperado;
    end
    secuencia_binaria = strcat(bits_demapeados{:});% convierte cada celda en cadena y se concatena
    secuencia_bits = double(secuencia_binaria) - 48;

% Cálculo de la BER estimada en esta iteración

secuencia_bits_2 = double(secuencia);

[~,BER(i)]=symerr(secuencia_bits_2, secuencia_bits);

end


%% Recuperacion de la imagen
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


%% Grafica BER Teorica vs Estimada

berTheory = berawgn(EbNoVec,'qam',M);

figure;
semilogy(EbNoVec, berTheory, 'b-'); % Curva de BER teórica en azul
hold on;
semilogy(EbNoVec, BER, 'r-*'); % Curva de BER estimada en rojo con asteriscos
grid on;
legend('BER Teorica', 'BER Estimada');
xlabel('Eb/No (dB)');
ylabel('Bit Error Rate');
title('Comparación de BER Teórica y Estimada');


%% Funciones, etc

function [mapeo, Es] = mapeo_Simbolos(grupo_Bits)

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

% Cálculo de la energía del símbolo
A = cell2mat(diccionario_8PSK(:,2));  % Extrae los valores complejos de la constelación
Es = sum(abs(A).^2) / numel(A);  % Calcula la energía del símbolo
end

% Funcion para determinar el Espectro

function [X,f] = FourierT(x,fs)

X = fft(x);
X = fftshift(X);
X = abs(X);
f = (-length(x)/2:length(x)/2-1) * fs / length(x);
end
