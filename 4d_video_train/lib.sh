function nerfstudio {
    #!/bin/bash
    NERF_PORT=${NERF_PORT:7007}
    docker run \
        --gpus all \
        -u "$(id -u)" \
        -e "CUDA_LAUNCH_BLOCKING=1" \
        -e "TORCH_USE_CUDA_DSA=1" \
        -v "$PWD:/workspace/" \
        -v "$HOME/.cache/:/home/user/.cache/" \
        -p ${NERF_PORT}:7007 \
        --rm \
        --shm-size=32gb \
        dromni/nerfstudio:main $@
}

function mapper {
    frame=$1
    fnum=$(basename "$frame")
    sparse_dir="$fnum-sparse"
    mkdir $sparse_dir
    nerfstudio colmap mapper \
        --database_path "./database-$fnum.db" \
        --image_path "$frame" \
        --output_path "$sparse_dir"
}

function process_images {
    frame=$1
    fnum=$(basename "$frame")
    output="./output-$fnum"
    mkdir -p $output
    find $frame
    nerfstudio ns-process-data images \
        --verbose \
        --data "${frame}" \
        --output-dir "$output" \
        --matching-method exhaustive
        #--skip-image-processing \
        #--eval-data scene_buffer \
        #--refine-pixsfm \
        #--sfm-tool hloc \
}

function export-ply {
    config=$1
    nerfstudio ns-export gaussian-splat \
        --load-config "$config" \
        --output-dir output-splat
}

function extract_frame {
	frame=$1
	fnum=$(basename "$frame")
	nerfstudio colmap feature_extractor \
		--database_path "database-${fnum}.db" \
		--image_path "${frame}"	
}
