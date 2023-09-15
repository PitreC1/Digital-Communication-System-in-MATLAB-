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

% conformar pulsos
%pulsos_conformados = filter(pulso,1,secuencia_Sobremuestreada);
pulsos_real = conv(secuencia_Sobremuestreada_real, pulso);
pulsos_img = conv(secuencia_Sobremuestreada_img, pulso);

% Graficar el pulso conformador
figure(4),
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
figure(5),
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

%% Modulación Normal Pasa Banda

Rs = 10;
fs = mps*Rs;%Frecuencia de muestreo
ts = 1/fs;
Tb = 1;
fc = 2*Rs;%Frecuencia portadora


t= 0:(1/fs):(length(pulsos_real) * Tb - 1/fc)/fs;

X1prima = sqrt(2)*pulsos_real.*cos(2*pi*fc.*t);
X2prima = -sqrt(2)*pulsos_img.*sin(2*pi*fc.*t);
X = X1prima+X2prima; %Señal a transmitir

figure,
plot(t,X);
title("Señal a Transmitir")


%% Canal AWGN

M=16; %Orden de Modulación
ebno=1000000; %EbNo en veces

sigma=sqrt(Energia/(2*log2(M)*ebno));%determina la varianza de ruido
Z=sigma*randn(1,length(X));
Y=X+Z;
figure,
subplot(211),plot(X,'k'),grid on, title('Señal Modulada'),xlim([mps*50 mps*100])
subplot(212),plot(Y,'g'),grid on, title('Señal a la salida del canal'),xlim([mps*50 mps*100])