//! # AVIF image serializer (muxer)
//!
//! ## Usage
//!
//! 1. Compress pixels using an AV1 encoder, such as [rav1e](https://lib.rs/rav1e). [libaom](https://lib.rs/libaom-sys) works too.
//!
//! 2. Call `avif_serialize::serialize_to_vec(av1_data, None, width, height, 8)`
//!
//! See [cavif](https://github.com/kornelski/cavif-rs) for a complete implementation.

mod boxes;
pub mod constants;
mod writer;

use crate::boxes::*;
use arrayvec::ArrayVec;
use std::io;

/// Config for the serialization (allows setting advanced image properties).
///
/// See [`Aviffy::new`].
pub struct Aviffy {
    premultiplied_alpha: bool,
    colr: ColrBox,
    min_seq_profile: u8,
    chroma_subsampling: (bool, bool),
}

/// Makes an AVIF file given encoded AV1 data (create the data with [`rav1e`](https://lib.rs/rav1e))
///
/// `color_av1_data` is already-encoded AV1 image data for the color channels (YUV, RGB, etc.).
/// The color image MUST have been encoded without chroma subsampling AKA YUV444 (`Cs444` in `rav1e`)
/// AV1 handles full-res color so effortlessly, you should never need chroma subsampling ever again.
///
/// Optional `alpha_av1_data` is a monochrome image (`rav1e` calls it "YUV400"/`Cs400`) representing transparency.
/// Alpha adds a lot of header bloat, so don't specify it unless it's necessary.
///
/// `width`/`height` is image size in pixels. It must of course match the size of encoded image data.
/// `depth_bits` should be 8, 10 or 12, depending on how the image was encoded (typically 8).
///
/// Color and alpha must have the same dimensions and depth.
///
/// Data is written (streamed) to `into_output`.
pub fn serialize<W: io::Write>(into_output: W, color_av1_data: &[u8], alpha_av1_data: Option<&[u8]>, width: u32, height: u32, depth_bits: u8) -> io::Result<()> {
    Aviffy::new().write(into_output, color_av1_data, alpha_av1_data, width, height, depth_bits)
}

impl Aviffy {
    #[must_use]
    pub fn new() -> Self {
        Self {
            premultiplied_alpha: false,
            min_seq_profile: 1,
            chroma_subsampling: (false, false),
            colr: Default::default(),
        }
    }

    /// Set whether image's colorspace uses premultiplied alpha, i.e. RGB channels were multiplied by their alpha value,
    /// so that transparent areas are all black. Image decoders will be instructed to undo the premultiplication.
    ///
    /// Premultiplied alpha images usually compress better and tolerate heavier compression, but
    /// may not be supported correctly by less capable AVIF decoders.
    ///
    /// This just sets the configuration property. The pixel data must have already been processed before compression.
    pub fn premultiplied_alpha(&mut self, is_premultiplied: bool) -> &mut Self {
        self.premultiplied_alpha = is_premultiplied;
        self
    }

    /// If set, must match the AV1 color payload, and will result in `colr` box added to AVIF.
    /// Defaults to BT.601, because that's what Safari assumes when `colr` is missing.
    /// Other browsers are smart enough to read this from the AV1 payload instead.
    pub fn matrix_coefficients(&mut self, matrix_coefficients: constants::MatrixCoefficients) -> &mut Self {
        self.colr.matrix_coefficients = matrix_coefficients;
        self
    }

    /// If set, must match the AV1 color payload, and will result in `colr` box added to AVIF.
    /// Defaults to sRGB.
    pub fn transfer_characteristics(&mut self, transfer_characteristics: constants::TransferCharacteristics) -> &mut Self {
        self.colr.transfer_characteristics = transfer_characteristics;
        self
    }

    /// If set, must match the AV1 color payload, and will result in `colr` box added to AVIF.
    /// Defaults to sRGB/Rec.709.
    pub fn color_primaries(&mut self, color_primaries: constants::ColorPrimaries) -> &mut Self {
        self.colr.color_primaries = color_primaries;
        self
    }

    /// If set, must match the AV1 color payload, and will result in `colr` box added to AVIF.
    /// Defaults to full.
    pub fn full_color_range(&mut self, full_range: bool) -> &mut Self {
        self.colr.full_range_flag = full_range;
        self
    }

    /// Makes an AVIF file given encoded AV1 data (create the data with [`rav1e`](https://lib.rs/rav1e))
    ///
    /// `color_av1_data` is already-encoded AV1 image data for the color channels (YUV, RGB, etc.).
    /// The color image MUST have been encoded without chroma subsampling AKA YUV444 (`Cs444` in `rav1e`)
    /// AV1 handles full-res color so effortlessly, you should never need chroma subsampling ever again.
    ///
    /// Optional `alpha_av1_data` is a monochrome image (`rav1e` calls it "YUV400"/`Cs400`) representing transparency.
    /// Alpha adds a lot of header bloat, so don't specify it unless it's necessary.
    ///
    /// `width`/`height` is image size in pixels. It must of course match the size of encoded image data.
    /// `depth_bits` should be 8, 10 or 12, depending on how the image has been encoded in AV1.
    ///
    /// Color and alpha must have the same dimensions and depth.
    ///
    /// Data is written (streamed) to `into_output`.
    pub fn write<W: io::Write>(&self, into_output: W, color_av1_data: &[u8], alpha_av1_data: Option<&[u8]>, width: u32, height: u32, depth_bits: u8) -> io::Result<()> {
        self.make_boxes(color_av1_data, alpha_av1_data, width, height, depth_bits).write(into_output)
    }

    fn make_boxes<'data>(&self, color_av1_data: &'data [u8], alpha_av1_data: Option<&'data [u8]>, width: u32, height: u32, depth_bits: u8) -> AvifFile<'data> {
        let mut image_items = ArrayVec::new();
        let mut iloc_items = ArrayVec::new();
        let mut compatible_brands = ArrayVec::new();
        let mut ipma_entries = ArrayVec::new();
        let mut data_chunks = ArrayVec::new();
        let mut irefs = ArrayVec::new();
        let mut ipco = IpcoBox::new();
        let color_image_id = 1;
        let alpha_image_id = 2;
        const ESSENTIAL_BIT: u8 = 0x80;
        let color_depth_bits = depth_bits;
        let alpha_depth_bits = depth_bits; // Sadly, the spec requires these to match.

        image_items.push(InfeBox {
            id: color_image_id,
            typ: FourCC(*b"av01"),
            name: "",
        });
        let ispe_prop = ipco.push(IpcoProp::Ispe(IspeBox { width, height }));
        // This is redundant, but Chrome wants it, and checks that it matches :(
        let av1c_color_prop = ipco.push(IpcoProp::Av1C(Av1CBox {
            seq_profile: self.min_seq_profile.max(if color_depth_bits >= 12 { 2 } else { 0 }),
            seq_level_idx_0: 31,
            seq_tier_0: false,
            high_bitdepth: color_depth_bits >= 10,
            twelve_bit: color_depth_bits >= 12,
            monochrome: false,
            chroma_subsampling_x: self.chroma_subsampling.0,
            chroma_subsampling_y: self.chroma_subsampling.1,
            chroma_sample_position: 0,
        }));
        // Useless bloat
        let pixi_3 = ipco.push(IpcoProp::Pixi(PixiBox {
            channels: 3,
            depth: color_depth_bits,
        }));
        let mut prop_ids: ArrayVec<u8, 5> = [ispe_prop, av1c_color_prop | ESSENTIAL_BIT, pixi_3].into_iter().collect();
        // Redundant info, already in AV1
        if self.colr != Default::default() {
            let colr_color_prop = ipco.push(IpcoProp::Colr(self.colr));
            prop_ids.push(colr_color_prop);
        }
        ipma_entries.push(IpmaEntry {
            item_id: color_image_id,
            prop_ids,
        });

        if let Some(alpha_data) = alpha_av1_data {
            image_items.push(InfeBox {
                id: alpha_image_id,
                typ: FourCC(*b"av01"),
                name: "",
            });
            let av1c_alpha_prop = ipco.push(boxes::IpcoProp::Av1C(Av1CBox {
                seq_profile: if alpha_depth_bits >= 12 { 2 } else { 0 },
                seq_level_idx_0: 31,
                seq_tier_0: false,
                high_bitdepth: alpha_depth_bits >= 10,
                twelve_bit: alpha_depth_bits >= 12,
                monochrome: true,
                chroma_subsampling_x: true,
                chroma_subsampling_y: true,
                chroma_sample_position: 0,
            }));
            // So pointless
            let pixi_1 = ipco.push(IpcoProp::Pixi(PixiBox {
                channels: 1,
                depth: alpha_depth_bits,
            }));

            // that's a silly way to add 1 bit of information, isn't it?
            let auxc_prop = ipco.push(IpcoProp::AuxC(AuxCBox {
                urn: "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha",
            }));
            irefs.push(IrefBox {
                entry: IrefEntryBox {
                    from_id: alpha_image_id,
                    to_id: color_image_id,
                    typ: FourCC(*b"auxl"),
                },
            });
            if self.premultiplied_alpha {
                irefs.push(IrefBox {
                    entry: IrefEntryBox {
                        from_id: color_image_id,
                        to_id: alpha_image_id,
                        typ: FourCC(*b"prem"),
                    },
                });
            }
            ipma_entries.push(IpmaEntry {
                item_id: alpha_image_id,
                prop_ids: [ispe_prop, av1c_alpha_prop | ESSENTIAL_BIT, auxc_prop, pixi_1].into_iter().collect(),
            });

            // Use interleaved color and alpha, with alpha first.
            // Makes it possible to display partial image.
            iloc_items.push(IlocItem {
                id: color_image_id,
                extents: [
                    IlocExtent {
                        offset: IlocOffset::Relative(alpha_data.len()),
                        len: color_av1_data.len(),
                    },
                ].into(),
            });
            iloc_items.push(IlocItem {
                id: alpha_image_id,
                extents: [
                    IlocExtent {
                        offset: IlocOffset::Relative(0),
                        len: alpha_data.len(),
                    },
                ].into(),
            });
            data_chunks.push(alpha_data);
            data_chunks.push(color_av1_data);
        } else {
            iloc_items.push(IlocItem {
                id: color_image_id,
                extents: [
                    IlocExtent {
                        offset: IlocOffset::Relative(0),
                        len: color_av1_data.len(),
                    },
                ].into(),
            });
            data_chunks.push(color_av1_data);
        };

        compatible_brands.push(FourCC(*b"mif1"));
        compatible_brands.push(FourCC(*b"miaf"));
        AvifFile {
            ftyp: FtypBox {
                major_brand: FourCC(*b"avif"),
                minor_version: 0,
                compatible_brands,
            },
            meta: MetaBox {
                hdlr: HdlrBox {},
                iinf: IinfBox { items: image_items },
                pitm: PitmBox(color_image_id),
                iloc: IlocBox { items: iloc_items },
                iprp: IprpBox {
                    ipco,
                    // It's not enough to define these properties,
                    // they must be assigned to the image
                    ipma: IpmaBox {
                        entries: ipma_entries,
                    },
                },
                iref: irefs,
            },
            // Here's the actual data. If HEIF wasn't such a kitchen sink, this
            // would have been the only data this file needs.
            mdat: MdatBox {
                data_chunks,
            },
        }
    }

    #[must_use] pub fn to_vec(&self, color_av1_data: &[u8], alpha_av1_data: Option<&[u8]>, width: u32, height: u32, depth_bits: u8) -> Vec<u8> {
        let mut out = Vec::with_capacity(color_av1_data.len() + alpha_av1_data.map_or(0, |a| a.len()) + 410);
        self.write(&mut out, color_av1_data, alpha_av1_data, width, height, depth_bits).unwrap(); // Vec can't fail
        out
    }

    pub fn set_chroma_subsampling(&mut self, subsampled_xy: (bool, bool)) -> &mut Self {
        self.chroma_subsampling = subsampled_xy;
        self
    }

    pub fn set_seq_profile(&mut self, seq_profile: u8) -> &mut Self {
        self.min_seq_profile = seq_profile;
        self
    }
}

/// See [`serialize`] for description. This one makes a `Vec` instead of using `io::Write`.
#[must_use] pub fn serialize_to_vec(color_av1_data: &[u8], alpha_av1_data: Option<&[u8]>, width: u32, height: u32, depth_bits: u8) -> Vec<u8> {
    Aviffy::new().to_vec(color_av1_data, alpha_av1_data, width, height, depth_bits)
}

#[test]
fn test_roundtrip_parse_mp4() {
    let test_img = b"av12356abc";
    let avif = serialize_to_vec(test_img, None, 10, 20, 8);

    let ctx = mp4parse::read_avif(&mut avif.as_slice(), mp4parse::ParseStrictness::Normal).unwrap();

    assert_eq!(&test_img[..], ctx.primary_item_coded_data().unwrap());
}

#[test]
fn test_roundtrip_parse_mp4_alpha() {
    let test_img = b"av12356abc";
    let test_a = b"alpha";
    let avif = serialize_to_vec(test_img, Some(test_a), 10, 20, 8);

    let ctx = mp4parse::read_avif(&mut avif.as_slice(), mp4parse::ParseStrictness::Normal).unwrap();

    assert_eq!(&test_img[..], ctx.primary_item_coded_data().unwrap());
    assert_eq!(&test_a[..], ctx.alpha_item_coded_data().unwrap());
}

#[test]
fn test_roundtrip_parse_avif() {
    let test_img = [1,2,3,4,5,6];
    let test_alpha = [77,88,99];
    let avif = serialize_to_vec(&test_img, Some(&test_alpha), 10, 20, 8);

    let ctx = avif_parse::read_avif(&mut avif.as_slice()).unwrap();

    assert_eq!(&test_img[..], ctx.primary_item.as_slice());
    assert_eq!(&test_alpha[..], ctx.alpha_item.as_deref().unwrap());
}

#[test]
fn test_roundtrip_parse_avif_colr() {
    let test_img = [1,2,3,4,5,6];
    let test_alpha = [77,88,99];
    let avif = Aviffy::new()
        .matrix_coefficients(constants::MatrixCoefficients::Bt709)
        .to_vec(&test_img, Some(&test_alpha), 10, 20, 8);

    let ctx = avif_parse::read_avif(&mut avif.as_slice()).unwrap();

    assert_eq!(&test_img[..], ctx.primary_item.as_slice());
    assert_eq!(&test_alpha[..], ctx.alpha_item.as_deref().unwrap());
}

#[test]
fn premultiplied_flag() {
    let test_img = [1,2,3,4];
    let test_alpha = [55,66,77,88,99];
    let avif = Aviffy::new().premultiplied_alpha(true).to_vec(&test_img, Some(&test_alpha), 5, 5, 8);

    let ctx = avif_parse::read_avif(&mut avif.as_slice()).unwrap();

    assert!(ctx.premultiplied_alpha);
    assert_eq!(&test_img[..], ctx.primary_item.as_slice());
    assert_eq!(&test_alpha[..], ctx.alpha_item.as_deref().unwrap());
}
