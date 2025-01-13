use cubecl_core as cubecl;
use cubecl_core::{prelude::*, CubeType};

use super::{
    base::{BatchOffsets, Coordinates, Dimensions, SharedMemories},
    config::CubeTiling2dConfig,
    tile::block_io::{
        base::BlockLoader, horizontal_block_check::HorizontalCheckBlockIO,
        unchecked_block::UncheckedBlockIO, vertical_block_check::VerticalCheckBlockIO,
        whole_block_check::WholeCheckBlockIO,
    },
};

#[derive(CubeType)]
#[allow(dead_code)]
pub(crate) struct LoadInfo<F: Float> {
    pub coordinates: Coordinates,
    pub k: u32,
    pub batch_offset: u32,
    pub shared_memory: SharedMemory<Line<F>>,
    pub dims: Dimensions,
}

#[cube]
pub(crate) trait Loader<F: Float>: Sync + Send + 'static {
    fn load_lhs_plain<B: BlockLoader<F>>(
        lhs: &Tensor<Line<F>>,
        load_info: LoadInfo<F>,
        #[comptime] config: CubeTiling2dConfig,
    );
    fn load_lhs_transposed<B: BlockLoader<F>>(
        lhs: &Tensor<Line<F>>,
        load_info: LoadInfo<F>,
        #[comptime] config: CubeTiling2dConfig,
    );
    fn load_rhs_plain<B: BlockLoader<F>>(
        rhs: &Tensor<Line<F>>,
        load_info: LoadInfo<F>,
        #[comptime] config: CubeTiling2dConfig,
    );
    fn load_rhs_transposed<B: BlockLoader<F>>(
        rhs: &Tensor<Line<F>>,
        load_info: LoadInfo<F>,
        #[comptime] config: CubeTiling2dConfig,
    );
}

#[cube]
pub(crate) fn load_to_shared_memories<F: Float, L: Loader<F>>(
    lhs: &Tensor<Line<F>>,
    rhs: &Tensor<Line<F>>,
    coordinates: Coordinates,
    k: u32,
    offsets: BatchOffsets,
    shared: SharedMemories<F>,
    #[comptime] config: CubeTiling2dConfig,
    dims: Dimensions,
) {
    let lhs_transposed = config.lhs_transposed;
    let rhs_transposed = config.rhs_transposed;

    let lhs_load_info = LoadInfo::<F> {
        coordinates,
        k,
        batch_offset: offsets.lhs,
        shared_memory: shared.lhs,
        dims,
    };
    let rhs_load_info = LoadInfo::<F> {
        coordinates,
        k,
        batch_offset: offsets.rhs,
        shared_memory: shared.rhs,
        dims,
    };

    // Lhs must be loaded as transposed. If it already is transposed in global memory, we load as plain.
    if lhs_transposed {
        load_lhs_plain::<F, L>(lhs, lhs_load_info, config);
    } else {
        load_lhs_transposed::<F, L>(lhs, lhs_load_info, config);
    }

    // Rhs must be loaded as plain. If it is transposed in global memory, we transpose it back.
    if rhs_transposed {
        load_rhs_transposed::<F, L>(rhs, rhs_load_info, config);
    } else {
        load_rhs_plain::<F, L>(rhs, rhs_load_info, config);
    }
}

#[cube]
pub(crate) fn load_lhs_transposed<F: Float, L: Loader<F>>(
    lhs: &Tensor<Line<F>>,
    load_info: LoadInfo<F>,
    #[comptime] config: CubeTiling2dConfig,
) {
    let check_m_bounds = config.check_m_bounds;
    let check_k_bounds = config.check_k_bounds;

    if check_m_bounds {
        if check_k_bounds {
            L::load_lhs_transposed::<WholeCheckBlockIO>(lhs, load_info, config);
        } else {
            L::load_lhs_transposed::<VerticalCheckBlockIO>(lhs, load_info, config);
        }
    } else if check_k_bounds {
        L::load_lhs_transposed::<HorizontalCheckBlockIO>(lhs, load_info, config);
    } else {
        L::load_lhs_transposed::<UncheckedBlockIO>(lhs, load_info, config);
    }
}

#[cube]
pub(crate) fn load_lhs_plain<F: Float, L: Loader<F>>(
    lhs: &Tensor<Line<F>>,
    load_info: LoadInfo<F>,
    #[comptime] config: CubeTiling2dConfig,
) {
    let check_m_bounds = config.check_m_bounds;
    let check_k_bounds = config.check_k_bounds;

    if check_k_bounds {
        if check_m_bounds {
            L::load_lhs_plain::<WholeCheckBlockIO>(lhs, load_info, config);
        } else {
            L::load_lhs_plain::<VerticalCheckBlockIO>(lhs, load_info, config);
        }
    } else if check_m_bounds {
        L::load_lhs_plain::<HorizontalCheckBlockIO>(lhs, load_info, config);
    } else {
        L::load_lhs_plain::<UncheckedBlockIO>(lhs, load_info, config);
    }
}

#[cube]
pub(crate) fn load_rhs_transposed<F: Float, L: Loader<F>>(
    rhs: &Tensor<Line<F>>,
    load_info: LoadInfo<F>,
    #[comptime] config: CubeTiling2dConfig,
) {
    let check_k_bounds = config.check_k_bounds;
    let check_n_bounds = config.check_n_bounds;

    if check_n_bounds {
        if check_k_bounds {
            L::load_rhs_transposed::<WholeCheckBlockIO>(rhs, load_info, config);
        } else {
            L::load_rhs_transposed::<VerticalCheckBlockIO>(rhs, load_info, config);
        }
    } else if check_k_bounds {
        L::load_rhs_transposed::<HorizontalCheckBlockIO>(rhs, load_info, config);
    } else {
        L::load_rhs_transposed::<UncheckedBlockIO>(rhs, load_info, config);
    }
}

#[cube]
pub(crate) fn load_rhs_plain<F: Float, L: Loader<F>>(
    rhs: &Tensor<Line<F>>,
    load_info: LoadInfo<F>,
    #[comptime] config: CubeTiling2dConfig,
) {
    let check_k_bounds = config.check_k_bounds;
    let check_n_bounds = config.check_n_bounds;

    if check_k_bounds {
        if check_n_bounds {
            L::load_rhs_plain::<WholeCheckBlockIO>(rhs, load_info, config);
        } else {
            L::load_rhs_plain::<VerticalCheckBlockIO>(rhs, load_info, config);
        }
    } else if check_n_bounds {
        L::load_rhs_plain::<HorizontalCheckBlockIO>(rhs, load_info, config);
    } else {
        L::load_rhs_plain::<UncheckedBlockIO>(rhs, load_info, config);
    }
}
