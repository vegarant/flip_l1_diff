function hvec = geth(nodes)
    % Check if the input mesh has at least two nodes
    if numel(nodes) < 2
        error('Input mesh must have at least two nodes.');
    end

    % Initialize the refined mesh with the first node
    hvec = (nodes(2) + nodes(1));

    % Iterate over each pair of adjacent nodes and add a refined point
    for i = 2:numel(nodes)-1
        % Add the midpoint between the current and next node
        hvec = [hvec, (nodes(i+1) - nodes(i))];
    end

end



