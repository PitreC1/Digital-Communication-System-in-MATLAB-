
clear all;
close all;

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

%% ESPECTRO DE LA SEÑAL A TRANSMITIR

fs = 2*5*10^4;
senial = 2 * secuencia - 1;
[X,f] = FourierT(secuencia,fs);
figure(3)
plot(f, X);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal a transmitir');


%% MODULACION EN BANDA BASE 16 QAM


% Dividir la secuencia de bits en grupos de 4 y mapear a símbolos
grupo = 4;
%secuencia = [1,0,0,0,1,1,1,1,0,1,0,1,1,0,1,0];
numero_Simbolos = numel(secuencia) / grupo;
secuencia_Simbolos = zeros(1, numero_Simbolos);

for i = 1:numero_Simbolos
    indiceInicio = (i - 1) * grupo + 1;
    indiceFin = indiceInicio + grupo - 1;
    grupoBits = secuencia(indiceInicio:indiceFin);

    [secuencia_Simbolos(i),Energia] = mapeo_Simbolos(grupoBits);

end

secuencia_real = real(secuencia_Simbolos);
secuancia_img = imag(secuencia_Simbolos);

%Grafica de constelacion usando la secuencia de simbolos rectangulares
scatterplot(secuencia_Simbolos);
title('Diagrama de Constelación 16 QAM');
xlabel('Parte Real');
ylabel('Parte Imaginaria');
axis square;
grid on;

%% ESPECTRO DE LA SEÑAL A TRANSMITIR EN BANDA BASE 

fs = 2*5*10^4;
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
mps = 10; %muestras por simbolos

% Pulso conformador 
pulso = rcosdesign(alfa, span, mps, 'sqrt');

%sobremuestreo

secuencia_Sobremuestreada_real = upsample(secuencia_real, mps+1);
secuencia_Sobremuestreada_img = upsample(secuancia_img, mps+1);
secuencia_Sobremuestreada = upsample(secuencia_Simbolos, mps);

% conformar pulsos
%pulsos_conformados = filter(pulso,1,secuencia_Sobremuestreada);
pulsos_real = conv(secuencia_Sobremuestreada_real, pulso);
pulsos_img = conv(secuencia_Sobremuestreada_img, pulso);
pulsos_Conformados = conv(secuencia_Sobremuestreada, pulso);

% Graficar el pulso conformador
figure(6),
subplot(3,1,1)
stem(pulso);
title('Pulsos Conformador');
xlabel('Muestras');
ylabel('Amplitud');

subplot(3,1,2)
stem(secuencia_Sobremuestreada_real)
title('Secuencia sobremuestreada real');
xlabel('Muestras');
xlim([0,33]);
ylabel('Amplitud');

subplot(3,1,3)
stem(pulsos_real);
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
stem(secuencia_Sobremuestreada_img)
title('Secuencia sobremuestreada imaginaria');
xlabel('Muestras');
xlim([0,33]);
ylabel('Amplitud');

subplot(3,1,3)
stem(pulsos_img);
title('Señal Pulsos conformados imaginarios');
xlabel('Muestras');
xlim([0,33]);
ylabel('Amplitud');

%% ESPECTRO DE LA SEÑAL A LA SALIDA DEL PULSO CONFORMADOR

[X,f] = FourierT(pulsos_Conformados,fs);
figure(8)
plot(f, X);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal Modulada a la salida el Pulso conformador');


%% Modulación en fase y cuadratura Pasa Banda

Rs = 10^6;
fs = mps*Rs;%Frecuencia de muestreo
ts = 1/fs;
Tb = 1;
fc = 2*Rs;%Frecuencia portadora


t= 0:ts:(length(pulsos_real) * Tb - 1/fc)/fs;

X1prima = sqrt(2)*pulsos_real.*cos(2*pi*fc.*t);
X2prima = sqrt(2)*pulsos_img.*sin(2*pi*fc.*t);
X_Transmitir = X1prima - X2prima; %Señal a transmitir

figure(9),
plot(t,X_Transmitir);
title("Señal a Transmitir")
xlabel('Tiempo (s)');
ylabel('Amplitud');
grid on;

% ESPECTRO DE LA SEÑAL A TRANSMITIR EN PASA BANDA

[X,f] = FourierT(X_Transmitir,fs);
figure(10)
plot(f, X);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal Modulada en Pasa Banda');


%% CANAL AWGN

M=16; %Orden de Modulación
ebno=1000000; %EbNo en veces

sigma=sqrt(Energia/(2*log2(M)*ebno));%determina la varianza de ruido
Z=sigma*randn(1,length(X_Transmitir));
Y=X_Transmitir+Z;
figure(11),
subplot(211),plot(X_Transmitir),grid on, title('Señal Modulada'),xlim([mps*50 mps*200])
subplot(212),plot(Y,'b'),grid on, title('Señal a través de canal AWGN'),xlim([mps*50 mps*200])

%% MUTITRAYECTORIA

alpha1 = 0.6; % Constante de escala
alpha2 = 0.3;
t1 = 160; % Desplazamiento en el tiempo (segundos)
t2 = 350; 
 
Y_Multitrayecto_1 = alpha1*circshift(Y, [0, t1]);
Y_Multitrayecto_2 = alpha2*circshift(Y, [0, t2]); 

figure(12);
subplot(3,1,1);
plot(Y);
xlim([mps*50 mps*300]);
title('Señal Original x(t)');
xlabel('Tiempo (s)');
ylabel('Amplitud');

subplot(3,1,2);
plot(Y_Multitrayecto_1,'r');
xlim([mps*50 mps*300]);
title('Señal Trayecto 1');
xlabel('Tiempo (s)');
ylabel('Amplitud');
subplot(3,1,3);
plot(Y_Multitrayecto_2,'m');
xlim([mps*50 mps*300]);
title('Señal Trayecto 2');
xlabel('Tiempo (s)');
ylabel('Amplitud');

Y_Canal = Y + Y_Multitrayecto_1 + Y_Multitrayecto_2;
figure(13)
plot(Y_Canal)
xlim([mps*50 mps*300])
title('Señal a la Salida del Canal');

% ESPECTRO DE LA SEÑAL A LA SALIDA DEL CANAL 
[YAWGNM,f] = FourierT(Y_Canal,fs);
figure(14)
plot(f, YAWGNM);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal a la Salida del Canal');

%% Demodulacion Pasa Banda
senal_Y1 = sqrt(2)*Y_Canal.*cos(2*pi*fc.*t);
senal_Y2 = -sqrt(2)*Y_Canal.*sin(2*pi*fc.*t);

%% Filtro Acoplado
filtro_Y1 = filter(pulso,1,senal_Y1);
filtro_Y2 = filter(pulso,1,senal_Y2);
figure(15)
plot(filtro_Y1,filtro_Y2)

%% Demodulacion Banda Base
Y1 = downsample(filtro_Y1,mps +1);
Y2 = downsample(filtro_Y2,mps +1);

eyediagram([filtro_Y1(5000:55000)' filtro_Y2(5000:55000)'],2*fs)
title('Diagrama del ojo señal rx')
xlim([0,0.1]);
scatterplot(Y1+1j*Y2),title('simbolos recibidos')

%% Demapeo
num_simbolos_rx = numel(Y1);% Numero de sumbolor recibidos
simbolos_binarios = {'0000', '0001', '0011', '0010', '0100', '0101', '0111', '0110', '1110', '1111','1101','1100','1000','1001', '1011','1010'};
bits_demapeados = cell(1, num_simbolos_rx);% Vector para bits demapeados

 diccionario = {
        '0000', -3 + 3i;
        '0001', -1 + 3i;
        '0011', 1 + 3i;
        '0010', 3 + 3i;
        '0100', -3 + 1i;
        '0101', -1 + 1i;
        '0111', 1 + 1i;
        '0110', 3 + 1i;
        '1110', -3 - 1i;
        '1111', -1 - 1i;
        '1101', 1 - 1i;
        '1100', 3 - 1i;
        '1000', -3 - 3i;
        '1001', -1 - 3i;
        '1011', 1 - 3i;
        '1010', 3 - 3i
    };

% Para cada símbolo recibido se obtiene el simbolo complejo
for i = 1:num_simbolos_rx
    simbolo_rx = Y1(i) + 1j * Y2(i);
    distancia_minima = inf; % distancia minima
    grupo_bits_recuperado = [];

    % Para cada símbolo de la constelación
    for j = 1:M
        distancia = abs(simbolo_rx - diccionario{j, 2});% Calculo de la distancia para cada simbolo
        if distancia < distancia_minima
            distancia_minima = distancia;
            grupo_bits_recuperado = simbolos_binarios{j};
        end
    end
    bits_demapeados{i} = grupo_bits_recuperado;
end
secuencia_binaria = strcat(bits_demapeados{:});% convierte cada celda en cadena y se concatena
secuencia_bits = double(secuencia_binaria) - 48;
columnas_a_eliminar = [41401 41402 41403 41404]; % Índices de las columnas a eliminar
secuencia_bits(columnas_a_eliminar) = [];

%% Recuperacion de la imagen 
alto = 200;

% Calcular ancho en base a alto y la cantidad total de bits
ancho = numel(secuencia_bits) / alto;

% Asegurar que ancho sea un número entero
ancho = floor(ancho);

% Recalcular alto si es necesario para que coincida con la cantidad de bits
alto = floor((numel(secuencia_bits)) / ancho);

% Convertir la secuencia de bits en una matriz
matriz_bits = reshape(secuencia_bits, alto, ancho);

% Convertir la matriz de bits en una matriz de píxeles (escala de grises)
valores_pixeles = matriz_bits * 255;  % Suponiendo que 1 representa el blanco y 0 representa el negro

% Crear la imagen en escala de grises
imagen_escala_grises = uint8(valores_pixeles);  % Convertir a tipo de datos uint8 para crear la imagen

% Mostrar la imagen
figure(18)
imshow(imagen_escala_grises);
title('Imagen generada a partir de la secuencia de bits');

%% FUNCIONES

function [mapeo, Es] = mapeo_Simbolos(grupo_Bits)
    
    % Define un diccionario (matriz de celdas) para mapear los grupos de bits a valores

    diccionario = {
        '0000', -3 + 3i;
        '0001', -1 + 3i;
        '0011', 1 + 3i;
        '0010', 3 + 3i;
        '0100', -3 + 1i;
        '0101', -1 + 1i;
        '0111', 1 + 1i;
        '0110', 3 + 1i;
        '1110', -3 - 1i;
        '1111', -1 - 1i;
        '1101', 1 - 1i;
        '1100', 3 - 1i;
        '1000', -3 - 3i;
        '1001', -1 - 3i;
        '1011', 1 - 3i;
        '1010', 3 - 3i
    };

    % Convierte el grupo_Bits a una cadena
    grupo_Bits_str = sprintf('%d', grupo_Bits);

    % Busca el valor correspondiente en el diccionario
    index = find(strcmp(diccionario(:,1), grupo_Bits_str));
    
    if ~isempty(index)
        mapeo = diccionario{index, 2};
    else
        % En caso de que el grupo de bits no exista en el diccionario, puedes asignar un valor predeterminado o manejar el error según tus necesidades
        error('Grupo de bits no válido');
    end

    % Cálculo de la energía del símbolo
    A = cell2mat(diccionario(:,2));  % Extrae los valores complejos de la constelación
    Es = sum(abs(A).^2) / numel(A);  % Calcula la energía del símbolo


end


% Funcion para determinar el Espectro

function [X,f] = FourierT(x,fs)

    X = fft(x);
    X = fftshift(X);
    X = abs(X);
    f = (-length(x)/2:length(x)/2-1) * fs / length(x);

end

