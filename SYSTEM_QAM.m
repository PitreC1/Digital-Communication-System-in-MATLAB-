clear all 
close all
%% FUENTE BINARIA
img = imread("panda3.jpg");
coompa=15;

% Imagen original en escala de grises
img_Gray = rgb2gray(img);

% Imagen Binarizada
level = graythresh(img_Gray);
img_Binaria = im2bw(img_Gray, level);

% Secuencia de bits de la imagen
secuencia = reshape(img_Binaria, 1, []);

%Grafica de la Imagen Binarizada
figure,
imshow(img_Binaria);
title('Imagen Binarizada ');

% Espectro de la Señal de entrada
Timagen=size(secuencia,ndims(secuencia));   %Image period.
fi=1/Timagen; 
[X,f] = FourierT(secuencia,fi);
figure,
plot(f, X);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal de Entrada');


%% MODULACION EN BANDA BASE 16 QAM
%Parametros de la modulacion 
M=16;
Mpam=sqrt(M);
d=1;
A=zeros(1,Mpam);
for i=1:Mpam
    A(1,i)=(2*i-1-Mpam)*d;
end

%Energia del Simbolo
P=perms([A A]);%Todas las Combinaciones de A 
p2=reshape(P',[2,numel(P)/2])'; %Genera los pares {parte real, parte imaginaria}
p3=unique(p2,'rows'); %Elimina las repeticiones
dr=p3(:,1)'; di=p3(:,2)';
Es=sum(((dr.^2)+(di.^2))./M);

%Vectores de simbolos
grupo = 4;
numero_Simbolos = numel(secuencia) / grupo;
secuencia_Simbolos = zeros(1, numero_Simbolos);

for i = 1:numero_Simbolos
    indiceInicio = (i - 1) * grupo + 1;
    indiceFin = indiceInicio + grupo - 1;
    grupoBits = secuencia(indiceInicio:indiceFin);

    secuencia_Simbolos(i) = mapeo_Simbolos(grupoBits);

end

%Secuencia de simbolos reales e imaginarios
S1=real(secuencia_Simbolos);
S2=imag(secuencia_Simbolos);

%Grafica de los simbolos Transmitidos
scatterplot(secuencia_Simbolos), title('simbolos transmitidos')

%Grafica del Espectro en Modulacion QAM
fi = 2*5*10^4;
[X,f] = FourierT(secuencia_Simbolos,fi);
figure,
plot(f, X);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro Modulacion QAM');

%% PULSO CONFORMADOR
% Parámetros del pulso conformador
rolloff = 0.5; %factor de roll-off
span = 4; %numero de simbolos
mps = 6; %muestras por simbolos

% Pulso conformador 
pulso = rcosfir(rolloff, span, mps,1,'sqrt');

%Agregar ceros al final de la secuencia 
S1=[S1 zeros(1,2*span)];
S2=[S2 zeros(1,2*span)];

%Sobremuestreo
X1 = upsample(S1, mps);
X2= upsample(S2, mps);
X3= upsample(S1+j*S2,mps);

%Pulso Conformados
pulsos_conf_real = filter(pulso,1,X1);
pulsos_conf_img = filter(pulso,1,X2);
pulsos_conformados=filter(pulso,1,X3);

% Grafica del pulso conformador parte real 
figure,
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

% Grafica del pulso conformador parte imaginaria 
figure,
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

% ESPECTRO DE LA SEÑAL A LA SALIDA DEL PULSO CONFORMADOR

fi = (2*10^4)/3;
[XmBB,f] = FourierT(pulsos_conformados,fi);
figure,
plot(f, XmBB);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal a la salida del pulso conformador');

%% Modulacion Pasa Banda 
%Parametros de la modulacion
Rs=1;
fs=mps*Rs;
ts=1/fs;
Tb=1;
Ns=length(secuencia_Simbolos);
fc=2*Rs;

%Vector de tiempo
t=0:ts:(Ns+2*span)-ts;

%Señal Pasa Banda 
senal_real = sqrt(2)*pulsos_conf_real.*cos(2*pi*fc.*t);
senal_img = -sqrt(2)*pulsos_conf_img.*sin(2*pi*fc.*t);
senal_tx = senal_real + senal_img; 

% ESPECTRO DE LA SEÑAL MODULADA EN PASA BANDA
fi = (5*10^4)/3;
[XmBB,f] = FourierT(senal_tx,fi);
figure,
plot(f, XmBB);
xlabel('Frecuencia (Hz)');
ylabel('Amplitud');
title('Espectro de la señal Modulada en Pasa Banda');


%% Canal AWGN
M=16; %Orden de Modulación
EbNoVec = (0:10);

% Inicializa vectores para bits transmitidos y recibidos

bits_transmitidos_total = [];
bits_recibidos_total = [];

for i1 = 1:1:length(EbNoVec)

    ebno = 10.^((EbNoVec(i1))/(10));
    sigma=sqrt(Es/(2*log2(M)*ebno));%determina la varianza de ruido
    Z=sigma*randn(1,length(senal_tx));
    Y=senal_tx+Z;
    
%% MUTITRAYECTORIA

%Parametros de la multitrayectoria
alpha1 = 0.1; % Constante de escala
t1 = 10; % Desplazamiento en el tiempo (segundos)

Y_Multitrayecto_1 = alpha1*interp1(t, Y, t - t1, 'linear', 0); % Interpolación lineal
Y_Canal = Y + Y_Multitrayecto_1;

if i1 == length(EbNoVec)
        figure,
        subplot(2,1,1);
        plot(Y);
        title('Señal Original x(t)');
        xlabel('Tiempo (s)');
        xlim([0,300]);
        ylabel('Amplitud');
    
        subplot(2,1,2);
        plot(Y_Multitrayecto_1,'r');
        title('Señal Trayecto 1');
        xlabel('Tiempo (s)');
        xlim([0,300]);
        ylabel('Amplitud');
end
if i1 == length(EbNoVec)
        figure,
        plot(Y_Canal)
        xlim([0, 300]);
        title('Señal a la Salida del Canal');

        % ESPECTRO DE LA SEÑAL A LA SALIDA DEL CANAL
        [YAWGNM,f] = FourierT(Y_Canal,fi);
        figure(11),
        plot(f, YAWGNM);
        xlabel('Frecuencia (Hz)');
        ylabel('Amplitud');
        title('Espectro de la señal a la Salida del Canal');
end

%% Ecualizador
Y_Ecualizado1 = (Y_Canal) - (alpha1*interp1(t, Y_Canal, t - t1, 'linear', 0)+(alpha1^2)*interp1(t, Y_Canal, t - 2*t1, 'linear', 0));
Y_Ecualizado2 = Y_Ecualizado1 + (alpha1^2)*interp1(t, Y_Canal, t - 2*t1, 'linear', 0)+(alpha1^3)*interp1(t, Y_Canal, t - 3*t1, 'linear', 0);
 if i1 == length(EbNoVec)
        figure,
        subplot(2,1,1);
        plot(Y)
        xlim([mps*5 mps*30])
        title('Señal con AWGN');
        subplot(2,1,2);
        plot(Y_Ecualizado2)
        xlim([mps*5 mps*30])
        title('Señal Ecualizada');

        % ESPECTRO DE LA SEÑAL A LA SALIDA DEL ECUALIZADOR
        [YAWGNM,f] = FourierT(Y_Ecualizado2,fi);
        figure,
        plot(f, YAWGNM);
        xlabel('Frecuencia (Hz)');
        ylabel('Amplitud');
        title('Espectro de la señal a la Salida del Canal');
end    
%% Demodulacion Pasa Banda
senal_Y1 = sqrt(2)*Y_Ecualizado2.*cos(2*pi*fc.*t);
senal_Y2 = -sqrt(2)*Y_Ecualizado2.*sin(2*pi*fc.*t);
senal_YDM = senal_Y1 + senal_Y2;

%% Filtro Acoplado
filtro_Y1 = filter(pulso,1,senal_Y1);
filtro_Y2 = filter(pulso,1,senal_Y2);
if i1 == length(EbNoVec)
        figure,
        plot(filtro_Y1,filtro_Y2)
end
%% Demodulacion Banda Base
Y1 = downsample(filtro_Y1,mps);
Y2 = downsample(filtro_Y2,mps);
if i1 == length(EbNoVec)
        %Graficas del diagrama de ojo 
        eyediagram([filtro_Y1(5000:55000)' filtro_Y2(5000:55000)'],4*fs)
        title('Diagrama del ojo señal rx')

        %Grafica simbolos Recibidos
        scatterplot(Y1+1j*Y2),title('simbolos recibidos en Demodulacion')
end

%% De-mapeo- Criterio de la Distancia Minima
T = Mpam - 1; % Número de umbrales de decisión
Tes = zeros(1, T);
for a = 1:T
    Tes(1, a) = (A(1, a) + A(1, 1 + a)) / 2;
end

Se1=Y1;
Se2=Y2;
S_estimados = zeros(size(Se1));

for k = 1:length(Se1)
    real_part = Se1(k);
    imag_part = Se2(k);
    
    if real_part >= 0
        if imag_part >= 0
            % Primer Cuadrante
            if real_part <= 2
                if imag_part <= 2
                    S_estimados(k) = 1 + 1j;
                else
                    S_estimados(k) = 1 + 3j;
                end
            else
                if imag_part <= 2
                    S_estimados(k) = 3 + 1j;
                else
                    S_estimados(k) = 3 + 3j;
                end
            end
        else
            % Cuarto Cuadrante
            if real_part <= 2
                if imag_part >= -2
                    S_estimados(k) = 1 - 1j;
                else
                    S_estimados(k) = 1 - 3j;
                end
            else
                if imag_part >= -2
                    S_estimados(k) = 3 - 1j;
                else
                    S_estimados(k) = 3 - 3j;
                end
            end
        end
    else
        if imag_part >= 0
            % Segundo Cuadrante
            if real_part >= -2
                if imag_part <= 2
                    S_estimados(k) = -1 + 1j;
                else
                    S_estimados(k) = -1 + 3j;
                end
            else
                if imag_part <= 2
                    S_estimados(k) = -3 + 1j;
                else
                    S_estimados(k) = -3 + 3j;
                end
            end
        else
            % Tercer Cuadrante
            if real_part >= -2
                if imag_part >= -2
                    S_estimados(k) = -1 - 1j;
                else
                    S_estimados(k) = -1 - 3j;
                end
            else
                if imag_part >= -2
                    S_estimados(k) = -3 - 1j;
                else
                    S_estimados(k) = -3 - 3j;
                end
            end
        end
    end
end

%Elimiacion de ceros al final de la secuencia
S_estimados=S_estimados(2*span+1:length(S_estimados));  

%Demapeo de simbolos
secuencia_bits = demapeo_Simbolo(S_estimados);

%Grafica de los simbolos demapeados
if i1 == length(EbNoVec)
  scatterplot(S_estimados),title('simbolos recibidos después del Demapeo')       
end

%% Recuperacion de la Imagen
secuencia_bits=double(secuencia_bits)-48;
alto = 200;
ancho = numel(secuencia_bits) / alto;
ancho = floor(ancho);
alto=numel(secuencia_bits) / ancho;
matriz_bits = reshape(secuencia_bits, [alto,ancho]);
valores_pixeles = matriz_bits * 255;  %1 representa el blanco y 0 representa el negro
imagen_escala_grises = uint8(valores_pixeles);  % Convertir a tipo de datos uint8 para crear la imagen

%BER Estimada
bits_transmitidos = double(secuencia'); % Secuencia de bits transmitidos
bits_recibidos = double(secuencia_bits'); % Secuencia de bits recibidos
bits_incorrectos = sum(bits_transmitidos ~= bits_recibidos);
[~,BER(i1)]=symerr(bits_transmitidos,bits_recibidos);   

end    

% Grafica de la Imagen Recuperada
figure,
imshow(imagen_escala_grises);
title('Imagen generada a partir de la secuencia de bits');
%% BER Teorica vs Estimada
berTheory = berawgn(EbNoVec,'qam',M);
figure,
semilogy(EbNoVec, berTheory, 'b-'); % Curva de BER teórica en azul
hold on;
semilogy(EbNoVec, BER, 'r-*'); % Curva de BER estimada en rojo con asteriscos
grid on;
legend('BER Teorica', 'BER Estimada');
xlabel('Eb/No (dB)');
ylabel('Bit Error Rate');
title('Comparación de BER Teórica y Estimada');

%% Funciones
function mapeo = mapeo_Simbolos(grupo_Bits)
    diccionario = {
        '0000', -3 + 3i;
        '0001', -3 + 1i;
        '0010', -3 - 1i;
        '0011', -3 - 3i;

        '0100', -1 + 3i;
        '0101', -1 + 1i;
        '0110', -1 - 1i;
        '0111', -1 - 3i;
        
        '1000', 1 + 3i;
        '1001', 1 + 1i;
        '1010', 1 - 1i
        '1011', 1 - 3i;

        '1100', 3 + 3i;
        '1101', 3 + 1i;
        '1110', 3 - 1i;
        '1111', 3 - 3i;
       
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

end


function grupo_Bits = demapeo_Simbolo(valor_complejo)
    
    % Define un diccionario (matriz de celdas) para mapear valores a grupos de bits
    diccionario_invertido = {
        -3 + 3i, '0000';
        -3 + 1i, '0001';
        -3 - 1i, '0010';
        -3 - 3i, '0011';

        -1 + 3i, '0100';
        -1 + 1i, '0101';
        -1 - 1i, '0110';
        -1 - 3i, '0111';

        1 + 3i, '1000';
        1 + 1i, '1001';
        1 - 1i, '1010';
        1 - 3i, '1011';

        3 + 3i, '1100';
        3 + 1i, '1101';
        3 - 1i, '1110';
        3 - 3i, '1111';
        };

    % Busca el grupo de bits correspondiente en el diccionario
    grupo_Bits = '';

    for i = 1:length(valor_complejo)
        index = find(cellfun(@(x) isequal(x, valor_complejo(i)), diccionario_invertido(:,1)));
        if ~isempty(index)
            grupo_Bits = [grupo_Bits, diccionario_invertido{index, 2}];
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
