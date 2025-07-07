#!/bin/bash

### Please modify the time intervals corresponding to different scenarios in line 77 of the `scene/gaussian_model.py` file.
python train_themal_stage1.py -s your_data_path -m your_output_path
cp -r your_output_path your_output_path1


TARGET_DIR='/home/yk98/Themal_TempGS-master_nodata/outputs/feicuiwan0707_1/s1_all_real_smooth_20250514_stage3'
FEATURE_DIR="$TARGET_DIR/fature_time_net"
POINT_CLOUD_DIR="$TARGET_DIR/point_cloud"

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory $TARGET_DIR does not exist!"
    exit 1
fi

# Function: Delete all folders except the specified one in a directory
delete_except() {
    local dir="$1"
    local except="$2"
    
    if [ ! -d "$dir" ]; then
        echo "Warning: Directory $dir does not exist, skipping..."
        return
    fi
    
    echo "Processing directory: $dir"
    cd "$dir" || return
    
    # Delete all folders except the specified one
    for item in *; do
        if [ -d "$item" ] && [ "$item" != "$except" ]; then
            echo "Deleting: $dir/$item"
            rm -rf "$item"
        fi
    done
    
    cd - >/dev/null  # Return to the original directory
}

# Delete all folders except iteration_1 in fature_time_net
delete_except "$FEATURE_DIR" "iteration_1"

# Delete all folders except iteration_10000 in point_cloud
delete_except "$POINT_CLOUD_DIR" "iteration_10000"

# Rename iteration_10000 to iteration_1 in point_cloud
if [ -d "$POINT_CLOUD_DIR/iteration_10000" ]; then
    echo "Renaming: $POINT_CLOUD_DIR/iteration_10000 -> $POINT_CLOUD_DIR/iteration_1"
    mv "$POINT_CLOUD_DIR/iteration_10000" "$POINT_CLOUD_DIR/iteration_1"
else
    echo "Warning: $POINT_CLOUD_DIR/iteration_10000 does not exist, cannot rename!"
fi

echo "Operation completed!"

python train_themal_stage2.py -s your_data_path -m your_output_path1