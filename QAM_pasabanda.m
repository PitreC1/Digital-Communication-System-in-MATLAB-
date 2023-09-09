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
numero_Simbolos = numel(secuencia) / grupo;
secuencia_Simbolos = zeros(1, numero_Simbolos);

for i = 1:numero_Simbolos
    indiceInicio = (i - 1) * grupo + 1;
    indiceFin = indiceInicio + grupo - 1;
    grupoBits = secuencia(indiceInicio:indiceFin);

    secuencia_Simbolos(i) = mapeo_Simbolos(grupoBits);

end

secuencia_real = real(secuencia_Simbolos);
secuancia_img = imag(secuencia_Simbolos);

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
figure;
stem(pulso);
title('Pulso');
xlabel('Muestras');
ylabel('Amplitud');

% figure;
% plot(pulsos_real);
% title("Parte Real Simbolos")
% 
% figure;
% plot(pulsos_img);
% title("Parte Img Simbolos")

%% Modulación Normal Pasa Banda

Rs = 10;
fs = mps*Rs;%Frecuencia de muestreo
ts = 1/fs;
Tb = 1;
fc = 2*Rs;%Frecuencia portadora
c = mps + 1; %cantidad de ceros que toman los filtros p(t) y q(t)
ebno = 1000000; %EbNo en veces
% S1=[S1 zeros(1,2*c)];
% S2=[S2 zeros(1,2*c)];


t_fsk = 0:(1/fs):(length(pulsos_real) * Tb - 1/fc)/fs;

X1prima = sqrt(2)*pulsos_real.*cos(2*pi*fc.*t_fsk);
X2prima = -sqrt(2)*pulsos_img.*sin(2*pi*fc.*t_fsk);
X = X1prima+X2prima; %Señal a transmitir

figure,
plot(t_fsk,X);
title("Señal a Transmitir")
