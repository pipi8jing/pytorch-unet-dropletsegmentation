function U = Img2Ddenoise_chu(F, aTV, maxitr)


%   min aTV*TV(U) +  0.5|U - F|_2^2
%
% Inputs:
%
%  aTV, -- regularization parameters in the model (positive)
%  F,   -- measurement

    maxItr = 100;       % max # of iterations
    gamma = 1.0;        % noisy choice = 1.0
    beta = 10; %100         % noisy choice = 10
    relchg_tol = 5e-4;  % stopping tolerance based on relative change
[m n]=size(F);
U = zeros(m,n);     % initial U. 
if (aTV <= 0); error('aTV must be strictly positive'); end; 

%% initialize constants
Ux = zeros(m,n); Uy = zeros(m,n); 
bx = zeros(m,n); by = zeros(m,n); 
Wx = zeros(m,n); Wy = zeros(m,n); 
H=[1];
if (mod(m,2)==0)
    padsize1=floor(m/2);
    padsize2=padsize1-1;
else
    padsize1=floor(m/2);
    padsize2=padsize1;
end

if (mod(n,2)==0)
    padsize3=floor(n/2);
    padsize4=padsize3-1;
else
    padsize3=floor(n/2);
    padsize4=padsize3;
end
H=padarray(H,[padsize1,padsize3],0,'pre');
H=padarray(H,[padsize2,padsize4],0,'post');
%H=padarray(H,[500,500],0,'both');
H=fftshift(fft(H));
HtH=abs(H).^2;
Denorm=fftshift(abs(psf2otf([1,-1],[m,n])).^2+abs(psf2otf([1;-1],[m,n])).^2);
Denorm = Denorm*aTV*beta+1;
FF=fft2(F);
Denorm=ifftshift(Denorm);
%% Main loop
for ii = 1:maxitr %maxItr
     %   W-d-subprolem
    % ----------------
    [Wx, Wy] = Compute_Wxy(Ux,Uy,bx,by,aTV*beta);
    
    % ----------------
    %   U-subprolem
    % ----------------
    rhs = Compute_rhs_DtU(Wx,Wy,bx,by,(aTV*beta)); 
    U = ifft2((FF + fftn(rhs))./Denorm);
    U=real(U); 
    U(U<0)=0;
    [Ux,Uy]=Compute_Uxy(U);
   
    % ------------------------------------------
    % Bregman update
    %
    bx = bx + gamma*(Ux - Wx);
    by = by + gamma*(Uy - Wy);

end % outer

end

function rhs = Compute_rhs_DtU(Wx,Wy,bx,by,tau);
rhs= tau*(DxtU(Wx-bx)+DytU(Wy-by));

    % compute D'_x(U)
    function dxtu = DxtU(U)
        dxtu = [U(:,end)-U(:, 1) U(:,1:end-1)-U(:,2:end)];
    end

    % compute D'_y(U)
    function dytu = DytU(U)
        dytu = [U(end,:)-U(1, :); U(1:end-1,:)-U(2:end,:)];
    end
    
end

function [Ux,Uy]=Compute_Uxy(U);
    [nuy,nux]=size(U);
Ux = [diff(U,1,2), U(:,1) - U(:,nux)]; 
Uy = [diff(U,1,1); U(1,:)-U(nuy,:)];
end

function [Wx, Wy] = Compute_Wxy(Ux,Uy,bx,by,tau);
UUx = Ux + bx; UUy = Uy + by; 
    V = sqrt(UUx .* conj(UUx) + UUy .* conj(UUy));
    V = max(V - tau, 0) ./ max(V,1e-10);
    Wx = V .* UUx; Wy = V .* UUy; 
end