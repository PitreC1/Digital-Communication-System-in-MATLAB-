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
