function refinedNodes = refine_mesh(nodes)
    % Check if the input mesh has at least two nodes
    if numel(nodes) < 2
        error('Input mesh must have at least two nodes.');
    end

    % Initialize the refined mesh with the first node
    refinedNodes = (nodes(1) + nodes(2))/2;

    % Iterate over each pair of adjacent nodes and add a refined point
    for i = 2:numel(nodes)-1
        % Add the midpoint between the current and next node
        refinedNodes = [refinedNodes, (nodes(i) + nodes(i+1))/2];
    end

    % Add the last node of the original mesh to the refined mesh
    refinedNodes = sort([refinedNodes, nodes]);
end
