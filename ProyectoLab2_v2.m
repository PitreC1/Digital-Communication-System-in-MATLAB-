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

alfabeto = [0,0,0% S1
    0,0,1% S2
    0,1,0% S3
    0,1,1% S4
    1,0,0% S5
    1,0,1% S6
    1,1,0% S7
    1,1,1]; % S8

Simbolos = [-0.707 - 0.707j,
    -1 + 0j,
    0 + 1j,
    -0.707 + 0.707j,
    0 - 1j,
    0.707 - 0.707j,
    +0.707 + 0.707j,
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
            secuencia_Simbolos(i) = j;
            break;
        end
    end
end

%Transformacion a secuencia de simbolos rectangulares

secuencia_simbolos_rect= zeros(1,numero_Simbolos);
for i = 1:numero_Simbolos
    posicion = secuencia_Simbolos(i);
    secuencia_simbolos_rect(i) = Simbolos(posicion);
end
% Grafica de constelacion usando la secuencia de simbolos rectangulares
scatterplot(secuencia_simbolos_rect)
title('Diagrama de Constelación 8PSK secuencia');
xlabel('Parte Real');
ylabel('Parte Imaginaria');
axis square;
grid on;
% Grafica constelación de símbolos
simbolo_const = exp(1i * pi/4 * (1:8))
scatterplot(simbolo_const);
title('Diagrama de Constelación 8PSK');
xlabel('Parte Real');
ylabel('Parte Imaginaria');
axis square;
grid on;

%disp(Simbolos_rect);

% Visualiza
% figure(4),
% plot(t, secuencia);
% xlabel('Tiempo SEÑAL BITS');
% ylabel('Valor');
%
% figure(5),
% plot(t, modulada, 'b');
% xlabel('Tiempo');
% ylabel('Amplitud modulada');

% Configuración de los ejes para una mejor visualización
%axis([0 t(end) -1.5 1.5]);


%% PULSO CONFORMADOR
