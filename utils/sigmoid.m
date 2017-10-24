function y = sigmoid(x, dzdy)
%SIGMOID computes sigmoid function
%   used for the hidden layer in the autoencoder in our codes
%
% Author: Rahaf Aljundi
%
% See the COPYING file.

y = 1 ./ (1 + exp(-x));


if nargin == 2 && ~isempty(dzdy)
    
    assert(all(size(x) == size(dzdy)));
    
    y = dzdy .* y .* (1 - y);
    
end

end

