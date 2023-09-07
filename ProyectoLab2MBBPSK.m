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


%% MODULACION EN BANDA BASE 8PSK

% Se define el alfabeto
alfabeto = [
    0,0,0;% S1
    0,0,1;% S2
    0,1,0;% S3
    0,1,1;% S4
    1,0,0;% S5
    1,0,1;% S6
    1,1,0;% S7
    1,1,1;]; % S8
% Se define los simbolos en forma rectangular 
Simbolos = [-0.707 - 0.707j;
    -1 + 0j;
    0 + 1j;
    -0.707 + 0.707j;
    0 - 1j;
    0.707 - 0.707j;
    +0.707 + 0.707j;
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
            secuencia_Simbolos(i) = j; %Vector de simbolos 
            break;
        end
    end
end

%Transformacion a secuencia de simbolos rectangulares
secuencia_simbolos_rect = Simbolos(secuencia_Simbolos); 

% secuencia_simbolos_rect= zeros(1,numero_Simbolos);
% for i = 1:numero_Simbolos
%     posicion = secuencia_Simbolos(i);
%     secuencia_simbolos_rect(i) = Simbolos(posicion); % Secuencia simbolos rect
% end

% Grafica de constelacion usando la secuencia de simbolos rectangulares
scatterplot(secuencia_simbolos_rect);
title('Diagrama de Constelación 8PSK');
xlabel('Parte Real');
ylabel('Parte Imaginaria');
axis square;
grid on;

%% PULSO CONFORMADOR
% Parámetros del pulso conformador
alfa = 0.5; %factor de roll-off
span = 6; %numero de simbolos
mps = 8; %muestras por simbolos

% Pulso conformador 
pulso = rcosdesign(alfa, span, mps, 'sqrt');

%sobremuestreo
secuencia_Sobremuestreada = upsample(secuencia_Simbolos, mps);% quitar +1

% conformar pulsos
%pulsos_conformados = filter(pulso,1,secuencia_Sobremuestreada);
pulsos_conformados = conv(secuencia_Sobremuestreada, pulso);


% Graficas
figure;
subplot(3,1,1)
stem(pulso);
title('Pulsos Conformador');
xlabel('Muestras');
ylabel('Amplitud');

subplot(3,1,2)
stem(secuencia_Sobremuestreada)
title('Secuencia sobremuestreada');
xlabel('Muestras');
ylabel('Amplitud');

subplot(3,1,3)
stem(pulsos_conformados);
title('Señal Pulsos conformados');
xlabel('Muestras');
ylabel('Amplitud');


%% Modulacion Pasa Banda  ASK
% Parámetros:
frp = 1000; % Frecuencia de la portadora 
Ap = 4; % Amplitud portadora
Tb = 1; % Duración de un bit en segundos

% Vector de tiempo
fs = 100 * frp; % Tasa de muestreo (debe ser mayor que el doble de la frecuencia de la portadora)
t = 0:(1/fs):(length(pulsos_conformados) * Tb - 1/fs)/100000; % Vector de tiempo

% Frecuencia de la portadora en radianes por segundo
w = 2 * pi * frp;

% Portadora
portadora = Ap * cos(w * t);

% Moduladora
moduladora=pulsos_conformados;

% Modulación en pasa banda 
ask=(1+(Ap*moduladora)).*(cos(w*t));

% Gráfica de la señal 
subplot(3,1,1)
plot(t, portadora);
title('Señal portadora');
xlabel('Tiempo (s)');
ylabel('Amplitud');
grid on; 

subplot(3,1,2)
plot(t, moduladora);
title('Señal moduladora');
xlabel('Tiempo (s)');
ylabel('Amplitud');
grid on; 

subplot(3,1,3)
plot(t, ask);
title('Señal Modulada');
xlabel('Tiempo (s)');
ylabel('Amplitud');
grid on; 

% Espectro de la señal:
figure;
fsenal = abs(fft(ask));
f = linspace(0, frp, length(fsenal));
plot(f, fsenal);
title('Espectro de la Señal Modulada en Banda Pasante');
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
grid on; 

%% Modulacion Pasa Banda PSK
% Parámetros para la modulación FSK
f1 = 100000; % Frecuencia para la primera portadora (Hz)
f2 = 50000; % Frecuencia para la segunda portadora (Hz)
ts= 1/f1;

% Vector de tiempo para la señal FSK
ts_fsk = ts / mps; % Período de muestreo para la señal FSK
t = 0:(1/fs):(length(pulsos_conformados) * Tb - 1/fs)/100000; % Vector de tiempo da 100448 sirve para portadora y moduladora 
t_fsk = 0:ts_fsk:(length(pulsos_conformados) * ts - ts_fsk); % Vector de tiempo FSK  da 883583 sirve para señal modulada

% Portadora
portadora_fsk = 3 * cos(w* t);

% Moduladora
moduladora_fsk = pulsos_conformados;

% Generación de la señal FSK
senal_modulada_fsk = zeros(1, length(t_fsk)); % Inicializa la señal FSK

% Modula la señal de acuerdo a la secuencia de pulsos conformados
for i = 1:length(pulsos_conformados)
    if pulsos_conformados(i) > 2
        % Si el valor es positivo, asigna la primera frecuencia
        senal_modulada_fsk((i - 1) * mps + 1:i * mps) = cos(2 * pi * f1 * t_fsk((i - 1) * mps + 1:i * mps));
    else
        % Si el valor es negativo o cero, asigna la segunda frecuencia
        senal_modulada_fsk((i - 1) * mps + 1:i * mps) = cos(2 * pi * f2 * t_fsk((i - 1) * mps + 1:i * mps));
    end
end

% Grafica la señal FSK
% Gráfica de la señal 
subplot(3,1,1)
plot(t, portadora_fsk);
title('Señal portadora');
xlabel('Tiempo (s)');
ylabel('Amplitud');
grid on; 

subplot(3,1,2)
plot(t, moduladora_fsk);
title('Señal moduladora');
xlabel('Tiempo (s)');
ylabel('Amplitud');
grid on; 

subplot(3,1,3)
plot(t_fsk, senal_modulada_fsk);
title('Señal Modulada FSK (Frecuencia Variable)');
xlabel('Tiempo (s)');
ylabel('Amplitud');
grid on;










