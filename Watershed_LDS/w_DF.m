clc; clear; close all;

p = imread('data/DFM-ROI1-8bit.tif');
p2 = imread('data/_1118C1_DFM-ROI1.tif');
ptv = Img2Ddenoise_chu(p,0.2,20);
p2tv = Img2Ddenoise_chu(p2,0.2,20);

t1 = p2tv(172:215,404:447);
t1 = mat2gray(t1-min(t1(:)));
t2 = p2tv(185:228,247:290);
t2 = mat2gray(t2-min(t2(:)));
t3 = (strel('disk',15).Neighborhood)-0.7*padarray((strel('disk',12).Neighborhood),[3 3]);
figure(1);clf;
subplot(3,2,[1,3,5]);imshow(p2);
subplot(3,2,2);imshow(t1);
subplot(3,2,4);imshow(t2);
subplot(3,2,6);imshow(t3);

ts = {t1,t3};
for ij = 1:length(ts)
    te = ts{ij};
    Cx = normxcorr2(ts{ij},mat2gray(ptv));
    
    xidx = (ceil(size(te,1)/2)):(size(Cx,1)-(floor(size(te,1)/2)));
    yidx = (ceil(size(te,2)/2)):(size(Cx,2)-(floor(size(te,2)/2)));

    cxfgm(ij,:,:) = Cx(xidx,yidx);
    h = fspecial('unsharp');
    cxfgm(ij,:,:) = imfilter(squeeze(cxfgm(ij,:,:)),h);
    
    fgm(ij,:,:) = imextendedmax(squeeze(cxfgm(ij,:,:)),.07,8);
    L = bwlabel(squeeze(fgm(ij,:,:)));
    stats = regionprops(L,squeeze(cxfgm(ij,:,:)),'Area','MeanIntensity');
    statsI = regionprops(L,p,'MeanIntensity');
    ar = [stats.Area];
    mC = [stats.MeanIntensity];
    mI = [statsI.MeanIntensity];
    L2 = ismember(L,find(mI>35 & mC>0.2 & ar>3 & ar<100));
%     Lout = ismember(L,find(mC<0.2 | mI<20));
    cell(ij,:,:) = L2;
    
    figure(10+ij)
    clf
    subplot(1,3,1)
    imagesc(Cx(xidx,yidx));axis image;colormap gray;
    subplot(1,3,2)
    imshow(imoverlay(imoverlay(p,L>0,[1 0 0]),L2>0,[0 0.7 0]));
    subplot(1,3,3)
    plot3(ar,mI,mC,'bo')
    xlabel('area');ylabel('intensity');zlabel('correlation');grid on
end

fgmall = squeeze(any(cell>0,1));
figure(98)
clf
imshow(imoverlay(p,squeeze(cell(2,:,:)),[0 0.7 0]));

figure(99)
clf
imshow(imoverlay(p,fgmall,[0 0.7 0]))

fgm4 = squeeze(cell(2,:,:));

% Convert the image to grayscale
% I = rgb2gray(p);
figure(22)
I = uint8(ptv);
subplot(231); imshow(I,[]), title('Original image')

I2 = I;
% the maxima are superimposed on the I2 image
I2(fgm4) = 255;
subplot(232); imshow(I2), title('Regional maximums superimposed on the original image (I2)')

% creation of structuring element se2
% se2 = strel(ones(5,5));
% 
% % dilation of the maxima to avoid marker fragmentation
% fgm2 = imclose(fgm, se2);
% fgm3 = imerode(fgm2, se2);
% % remove markers smaller than 20 pixels from the image
% fgm4 = bwareaopen(fgm3, 4);
% 
% I3 = I;
% % the maximums now corrected are superimposed on image I3
% I3(fgm4) = 255;
% figure; subplot(231); imshow(I3), title('Regional maximums modified superimposed on the original image (fgm4)')

% graytresh: makes the Iobrcbr image binary using the greytresh function
Izach = I; Izach(I<35) = 0;
bw = imbinarize(Izach,'adaptive');
bw = fgm4;
subplot(234); imshow(bw), title('Thresholded opening-closing by reconstruction (bw)');

% for each pixel calculates its distance from the nearest non-zero value
D = bwdist(bw);
% extracts the skeleton of the zones of influence of the objects in the foreground.
% serves to limit the expansion of the watershed algorithm to the pass
% following.
DL = watershed(D);

% background markers
bgm = DL == 0;
subplot(235); imshow(bgm), title('Watershed ridge lines (bgm)')

% what used to be the regional highs are now the lows: that's what it is
% associated with the skeleton of the areas of influence.
gradmag2 = imimposemin(I, fgm4);
% subplot(233);imshow(gradmag2),title('gradmag2s')
%subplot(233);imshow(gradmag,[]),title('gradmag')

% runs the watershed algorithm to extract the final contours of the
% marker
L = watershed(gradmag2);
subplot(233);imshow(label2rgb(L)),title('watershed');
I4 = I;

% traces the outlines of foreground objects
% I4(imdilate(L == 0, ones(3, 3)) | bgm | fgm4) = 255;
% subplot(236), imshow(I4), title('Markers and object boundaries superimposed on original image (I4)')

% regions are colored
Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
% subplot(235), imshow(Lrgb)
% title('Colored watershed label matrix (Lrgb)')
% 
subplot(236), imshow(I), hold on
himage = imshow(Lrgb);
% the alpha channel allows you to make color shades transparent
% of the Lrgb image
set(himage, 'AlphaData', 0.3);
title('Lrgb superimposed transparently on original image')