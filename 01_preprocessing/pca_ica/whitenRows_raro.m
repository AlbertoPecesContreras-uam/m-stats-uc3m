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
R = cov(Z');

% Whiten data using SVD
[U, S, ~] = svd(R, 'econ');
autovalores = diag(S);

% --- EL PARCHE PARA EEG ---
% Calculamos un umbral de tolerancia. Todo autovalor que sea menor a, 
% por ejemplo, 1e-6 veces el autovalor más grande, se considera "ruido cero".
tolerancia = max(autovalores) * 1e-6;
componentes_validas = autovalores > tolerancia;

% Nos quedamos SOLO con los autovectores y autovalores seguros
U_seguro = U(:, componentes_validas);
autovalores_seguros = autovalores(componentes_validas);

% Realizamos el blanqueo (tipo PCA Whitening) solo con las dimensiones útiles.
% Nota: Al hacer esto, la dimensión de Zw será (Componentes x N) en lugar de (20 x N).
% Esto es exactamente lo que FastICA necesita para converger sin marearse.
T  = diag(1 ./ sqrt(autovalores_seguros)) * U_seguro';
Zw = T * Z;
end