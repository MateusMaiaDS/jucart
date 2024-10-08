mutable struct node

    index::Int64
    left::Union{Nothing,node}
    right::Union{Nothing,node}

    node(index::Int64) = new(index,nothing,nothing)
end

node_example = node(1)

print(node_example.index)
print(node_example.left)
print(node_example.right)

