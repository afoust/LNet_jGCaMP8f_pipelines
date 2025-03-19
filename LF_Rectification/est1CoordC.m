%This function estimates one coordinate of the center. The coordinate is measure along
%the vector that is orthogonal to vin;
%This function estimates the centers of nPnts points around the point centImg. These points lie along a line
%defined by the vin vectDirIn.
%vectOrmean is the coordinate of the center of the image measured from the usual origin of the image (top left).
%This coordinate is masured along a vector orthogonal to vectF
%Treas is  treashold that binarize the input image.

function [b, vectF, centersEstCoor] = est1CoordC(centImg, vin, nPnts, inpIMg, treas)
    % Initialize visualization flag to 0 (off).
    figOn = 0;

    % Estimate centers along half-lines in opposite directions.
    centersEst1 = estCentHalfLine(centImg, -vin, (nPnts + 1) / 2, inpIMg, treas);
    centersEst2 = estCentHalfLine(centImg, vin, (nPnts + 1) / 2, inpIMg, treas);

    % Combine and order the estimated centers from both half-lines.
    centersEst = [centersEst1(end:-1:1, :); centersEst2(2:end, :)];

    % Linear regression to find best fitting line through centers.
    A = [centersEst(:, 2), ones(size(centersEst, 1), 1)];
    b = A \ centersEst(:, 1);

    % Calculate perpendicular distance from points to the best fitting line.
    tmpCen = abs((centersEst(:, 2) * b(1) + b(2) - centersEst(:, 1))) ./ sqrt(centersEst(:, 2).^2 + 1);
    trsPerc = 0.8;
    
    % Filter points based on distance to refine the fit.
    tmpCen2(:, 1) = centersEst((tmpCen < quantile(tmpCen, trsPerc)), 1);
    tmpCen2(:, 2) = centersEst((tmpCen < quantile(tmpCen, trsPerc)), 2);

    % Refine linear regression with filtered points.
    A = [tmpCen2(:, 2), ones(size(tmpCen2, 1), 1)];
    b = A \ tmpCen2(:, 1);

    % Visualization of the fitting line and original points.
    if figOn == 1
        figure; plot(centersEst(:, 2), centersEst(:, 2) * b(1) + b(2));
        hold on; plot(centersEst(:, 2), centersEst(:, 1));
    end

    % Calculate direction vector (vectF) and its orthogonal (vectFOrt).
    if abs(b(1)) < 1
        vectF = [b(1), 1];
        vectF = vectF / norm(vectF);
        vectFOrt = [vectF(2), -vectF(1)];
    else
        vectF = [1, 1 / b(1)];
        vectF = vectF / norm(vectF);
        vectFOrt = [-vectF(2), vectF(1)];
    end

    % % Project centers onto the orthogonal vector and calculate the mean.
    % result1Orth = (centersEst) * vectFOrt';
    % result1Orth0mn = abs(result1Orth - mean(result1Orth));
    % vectOrmean = mean(result1Orth(result1Orth0mn < quantile(result1Orth0mn, 0.9)));

    % Project centers onto the direction vector.
    centersEstCoor = (centersEst) * vectF';
end
