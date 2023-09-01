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


%% PULSO CONFORMADOR
% Parámetros del pulso conformador
alfa = 0.5; %factor de roll-off
span = 1; %numero de simbolos
mps = 10; %muestras por simbolos

% Pulso conformador 
pulso = rcosdesign(alfa, span, mps, 'sqrt');

%sobremuestreo

secuencia_Sobremuestreada = upsample(secuencia_Simbolos, mps+1);

% conrmar pulsos
pulsos_conformados = filter(pulso,1,secuencia_Sobremuestreada);

% Graficar el pulso conformador
figure;
stem(pulso);
title('Pulso');
xlabel('Muestras');
ylabel('Amplitud');

figure;
plot(real(pulsos_conformados));