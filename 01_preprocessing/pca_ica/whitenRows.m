function [Zw, T] = whitenRows(Z)
%
% Syntax:       [Zw, T] = whitenRows(Z);
%               
% Inputs:       Z is an (d x n) matrix containing n samples of a
%               d-dimensional random vector. (20 x n)
%               
% Outputs:      Zw is the whitened version of Z
%               
%               T is the (d x d) whitening transformation of Z
%               
% Description:  Returns the whitened (identity covariance) version of the
%               input data
%               
% Notes:        (a) Must have n >= d to fully whitenRows Z
%               
%               (b) Z = T \ Zcw
%               
% Author:       Brian Moore
%               brimoor@umich.edu
%               
% Date:         November 1, 2016
%

% Compute sample covariance
R = cov(Z');% (20x20)

% Whiten data
[U, S, ~] = svd(R,'econ');

% S: matriz diagonal de autovalores
% U: matriz de autovectores (cada columna es un autovector)

% Un autovalor es un escalar que refleja cuánto ha cambiado la longitud
% de su autovector asociado al proyectar los datos sobre él. Si el 
% autovalor es mayor que 1, el autovector se estira. Si es menor que 1, el
% autovector se comprime. Si es negativo, el autovector se invierte y se
% escala según el valor absoluto del autovalor. Cuando proyectamos un 
% conjunto de datos X sobre un autovector, estamos "alineando" esos datos
% a lo largo de la dirección del autovector. El autovalor nos dice cuánto
% se estira o se comprime esa proyección.

% La siguiente relación solo ocurre en casos específicos, como en matrices
% de covarianza, pero no es una regla general. Cuando trabajamos con la
% matriz de covarianza y extraemos autovectores y autovalores, los
% autovectores reflejan las direcciones principales de variabilidad en los
% datos, mientras que los autovalores indican la magnitud de la varianza en
% las direcciones.

% Si tenemos un conjunto de datos X, calculamos la matriz de covarianza,
% extraemos matriz de autovectores y autovalores asociados, proyectamos X
% sobre un autovector cualquiera (el primero, por ejemplo), tendremos una
% distribución de los datos en esa dimensión/dirección. Si yo calculo la
% varianza de esos datos en esa dimensión, ese valor coincidirá con el
% autovalor asociado a ese autovector.

% Si tenemos un conjunto de datos X y normalizamos sus características
% siguiendo una Z-score, tendremos una matriz cuyas dimensiones poseen
% media 0 y varianza 1. A partir de X_norm, se calcula su matriz de
% covarianzas y se extrae matriz de autovectores y autovalores. Si yo
% proyecto X_norm sobre las nuevas componentes principales, obtendré una
% matriz Z_pca, cuyas dimensiones no tienen por qué poseer varianza
% unitaria. Esto sucede porque cada componente principal o autovector sobre
% el que se proyecta tiene una varianza asociada que es el autovalor.

% Para asegurarnos de que las dimensiones de Z_pca posean varianza 1,
% dividimos las componentes entre la raíz de su autovalor. La varianza de
% las columnas de Z_pca son los autovalores asociados a cada autovector.
% ¿Qué sentido tiene? Al dividir por los autovalores, aseguro que los datos
% proyectados en cada dirección estén en las mismas escalas. Consistencia.

% ¿Qué sentido tiene multiplicar por U'? Estamos regresando al espacio 
% original de características, pero nuestros datos están decorrelacionados 
% y con varianza unitaria. Es decir, estamos realizando una transformación 
% de regreso al espacio original, pero manteniendo las propiedades de
% decorrelación y varianza unitaria.

% En resumen, aplicamos PCA y dividimos las columnas de Z_pca por su 
% desviacion estándar, después regresamos a un espacio en el que las 
% dimensiones están decorrelacionadas y con varianza unitaria 
% multiplicando por U'.
T  = U * diag(1 ./ sqrt(diag(S))) * U';
Zw = T * Z;
