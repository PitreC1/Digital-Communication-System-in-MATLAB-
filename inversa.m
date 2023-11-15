%Crea el Canal MIMO
numtx = 2;
sp = 0.45;
txarraypos = (0:numtx-1)*sp;
numrx = 2;
rxarraypos = 300 + (0:numrx-1)*sp;
numscat = 2;

%Canal MIMO
chmat = scatteringchanmtx(txarraypos,rxarraypos,numscat);
chamat_inv=calcular_Inversa(chmat)
final=chamat_inv*chmat


function inversa = calcular_Inversa(matriz)
    % Verificar si la matriz es 2x2
    [filas, columnas] = size(matriz);
    if filas ~= 2 || columnas ~= 2
        error('La matriz debe ser de tamaño 2x2.');
    end

    % Extraer elementos de la matriz
    a = matriz(1,1);
    b = matriz(1,2);
    c = matriz(2,1);
    d = matriz(2,2);

    % Calcular el determinante
    determinante = a*d - b*c;

    % Verificar si la matriz es invertible (determinante no es cero)
    if abs(determinante) < eps
        error('La matriz no es invertible (determinante igual a cero).');
    end

    % Calcular la inversa
    inversa = (1/determinante) * [d, -b; -c, a];
end