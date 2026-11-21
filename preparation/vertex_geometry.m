function [vertexArea, meanCurvature, gaussianCurvature] = ...
    vertex_geometry(vertices, faces)

vertices = double(vertices);
faces = double(faces);
nVertices = size(vertices, 1);
first = faces(:, 1); second = faces(:, 2); third = faces(:, 3);

edge12 = vertices(second, :) - vertices(first, :);
edge13 = vertices(third, :) - vertices(first, :);
faceArea = 0.5 * sqrt(sum(cross(edge12, edge13, 2).^2, 2));

angleFirst = local_angle(vertices(second, :) - vertices(first, :), ...
    vertices(third, :) - vertices(first, :));
angleSecond = local_angle(vertices(third, :) - vertices(second, :), ...
    vertices(first, :) - vertices(second, :));
angleThird = pi - angleFirst - angleSecond;

vertexArea = accumarray([first; second; third], ...
    repmat(faceArea, 3, 1) / 3, [nVertices, 1]);
vertexArea = max(vertexArea, eps);

angleSum = accumarray([first; second; third], ...
    [angleFirst; angleSecond; angleThird], [nVertices, 1]);
gaussianCurvature = abs((2*pi - angleSum) ./ vertexArea);

cotFirst = 1 ./ tan(angleFirst);
cotSecond = 1 ./ tan(angleSecond);
cotThird = 1 ./ tan(angleThird);
row = [second; third; third; first; first; second];
column = [third; second; first; third; second; first];
weight = [cotFirst; cotFirst; cotSecond; cotSecond; cotThird; cotThird];
adjacency = sparse(row, column, weight, nVertices, nVertices);
laplacian = spdiags(sum(adjacency, 2), 0, nVertices, nVertices) - adjacency;
coordinateLaplacian = laplacian * vertices;
meanCurvature = sqrt(sum(coordinateLaplacian.^2, 2)) ./ (4 * vertexArea);
end


function angle = local_angle(first, second)
crossNorm = sqrt(sum(cross(first, second, 2).^2, 2));
dotProduct = sum(first .* second, 2);
angle = atan2(crossNorm, dotProduct);
end

