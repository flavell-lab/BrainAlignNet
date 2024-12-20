using Random
using Images
using ImageFiltering

function adjust_image_cm(image, center, target_dim)

    img_adj = zeros(UInt16, target_dim)
    dim_x, dim_y, dim_z = size(image)
    center_x, center_y, center_z = center
    dx, dy, dz = round.(Int, target_dim ./ 2)

    rg_x = max(center_x - dx + 1, 1):min(center_x + dx, dim_x)
    rg_y = max(center_y - dy + 1, 1):min(center_y + dy, dim_y)
    rg_z = max(center_z - dz + 1, 1):min(center_z + dz, dim_z)

    rg_x_ = round(Int, dx - length(rg_x) / 2 + 1):round(Int, dx + length(rg_x) / 2)
    rg_y_ = round(Int, dy - length(rg_y) / 2 + 1):round(Int, dy + length(rg_y) / 2)
    rg_z_ = round(Int, dz - length(rg_z) / 2 + 1):round(Int, dz + length(rg_z) / 2)

    img_adj[rg_x_, rg_y_, rg_z_] .= image[rg_x, rg_y, rg_z]
    return img_adj
end
