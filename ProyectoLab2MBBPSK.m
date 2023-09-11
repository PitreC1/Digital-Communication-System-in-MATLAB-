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
secuencia_simbolos_real= real(secuencia_simbolos_rect);
secuecia_simbolos_img= imag(secuencia_simbolos_rect);

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
span = 1; %numero de simbolos
mps = 2; %muestras por simbolos

% Pulso conformador 
pulso = rcosdesign(alfa, span, mps, 'sqrt');

%sobremuestreo
secuencia_Sobremuestreada_r = upsample(secuencia_simbolos_real, mps+1);
secuencia_Sobremuestreada_i = upsample(secuecia_simbolos_img, mps+1);
% conformar pulsos
pulsos_conf_real = conv(secuencia_Sobremuestreada_r, pulso);
pulsos_conf_img = conv(secuencia_Sobremuestreada_i, pulso);

% Graficas
figure(4),
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
figure(5),
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

%% Modulacion Pasa Banda 
Rs=10;
fs=mps*Rs;
ts=1/fs;
Tb=1;
fc=2*Rs;
t = 0:ts:(length(pulsos_conf_real)*Tb - 1/fc)/fs;
% Moduladora
moduladora_PB=pulsos_conf_real;
senal_real=sqrt(2)*pulsos_conf_img.*cos(2*pi*fc.*t);
senal_img=sqrt(2)*pulsos_conf_img.*sin(2*pi*fc.*t);
senal_tx=senal_real - senal_img; %Señal a transmitir

%Graficas
figure(7),
plot(t, moduladora_PB);
title('Señal a transmitir');
xlabel('Tiempo (s)');
xlim([0,33]);
ylabel('Amplitud');
grid on; 
figure(8),
plot(t, senal_tx);
title('Señal a transmitir');
xlabel('Tiempo (s)');
xlim([0,33]);
ylabel('Amplitud');
grid on; 
