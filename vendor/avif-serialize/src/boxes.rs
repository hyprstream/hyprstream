use crate::constants::ColorPrimaries;
use crate::constants::MatrixCoefficients;
use crate::constants::TransferCharacteristics;
use crate::writer::Writer;
use crate::writer::WriterBackend;
use crate::writer::IO;
use arrayvec::ArrayVec;
use std::fmt;
use std::io;
use std::io::Write;

pub trait MpegBox {
    fn len(&self) -> usize;
    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error>;
}

#[derive(Copy, Clone)]
pub struct FourCC(pub [u8; 4]);

impl fmt::Debug for FourCC {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match std::str::from_utf8(&self.0) {
            Ok(s) => s.fmt(f),
            Err(_) => self.0.fmt(f),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AvifFile<'data> {
    pub ftyp: FtypBox,
    pub meta: MetaBox,
    pub mdat: MdatBox<'data>,
}

impl AvifFile<'_> {
    /// Where the primary data starts inside the `mdat` box, for `iloc`'s offset
    fn mdat_payload_start_offset(&self) -> u32 {
        (self.ftyp.len() + self.meta.len()
            + BASIC_BOX_SIZE) as u32 // mdat head
    }

    /// `iloc` is mostly unnecssary, high risk of out-of-buffer accesses in parsers that don't pay attention,
    /// and also awkward to serialize, because its content depends on its own serialized byte size.
    fn fix_iloc_positions(&mut self) {
        let start_offset = self.mdat_payload_start_offset();
        for iloc_item in self.meta.iloc.items.iter_mut() {
            for ex in iloc_item.extents.iter_mut() {
                let abs = match ex.offset {
                    IlocOffset::Relative(n) => n as u32 + start_offset,
                    IlocOffset::Absolute(_) => continue,
                };
                ex.offset = IlocOffset::Absolute(abs);
            }
        }
    }

    pub fn write<W: Write>(&mut self, mut out: W) -> io::Result<()> {
        self.fix_iloc_positions();

        let mut tmp = Vec::with_capacity(self.ftyp.len() + self.meta.len());
        let mut w = Writer::new(&mut tmp);
        let _ = self.ftyp.write(&mut w);
        let _ = self.meta.write(&mut w);
        drop(w);
        out.write_all(&tmp)?;
        drop(tmp);

        let mut out = IO(out);
        let mut w = Writer::new(&mut out);
        self.mdat.write(&mut w)?;
        Ok(())
    }
}

const BASIC_BOX_SIZE: usize = 8;
const FULL_BOX_SIZE: usize = BASIC_BOX_SIZE + 4;

#[derive(Debug, Clone)]
pub struct FtypBox {
    pub major_brand: FourCC,
    pub minor_version: u32,
    pub compatible_brands: ArrayVec<FourCC, 2>,
}

/// File Type box (chunk)
impl MpegBox for FtypBox {
    #[inline(always)]
    fn len(&self) -> usize {
        BASIC_BOX_SIZE
        + 4 // brand
        + 4 // ver
        + 4 * self.compatible_brands.len()
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.basic_box(*b"ftyp")?;
        b.push(&self.major_brand.0)?;
        b.u32(self.minor_version)?;
        for cb in &self.compatible_brands {
            b.push(&cb.0)?;
        }
        Ok(())
    }
}

/// Metadata box
#[derive(Debug, Clone)]
pub struct MetaBox {
    pub hdlr: HdlrBox,
    pub iloc: IlocBox,
    pub iinf: IinfBox,
    pub pitm: PitmBox,
    pub iprp: IprpBox,
    pub iref: ArrayVec<IrefBox, 2>,
}

impl MpegBox for MetaBox {
    #[inline]
    fn len(&self) -> usize {
        FULL_BOX_SIZE
            + self.hdlr.len()
            + self.pitm.len()
            + self.iloc.len()
            + self.iinf.len()
            + self.iprp.len()
            + IrefBox2 {
                entries: self.iref.iter().map(|e| e.entry).collect(),
            }.len()
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.full_box(*b"meta", 0)?;
        self.hdlr.write(&mut b)?;
        self.pitm.write(&mut b)?;
        self.iloc.write(&mut b)?;
        self.iinf.write(&mut b)?;
        let iref_fixed = IrefBox2 {
            entries: self.iref.iter().map(|e| e.entry).collect(),
        };
        iref_fixed.write(&mut b)?;
        self.iprp.write(&mut b)
    }
}

/// Item Info box
#[derive(Debug, Clone)]
pub struct IinfBox {
    pub items: ArrayVec<InfeBox, 2>,
}

impl MpegBox for IinfBox {
    #[inline]
    fn len(&self) -> usize {
        FULL_BOX_SIZE
        + 2 // num items u16
        + self.items.iter().map(|item| item.len()).sum::<usize>()
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.full_box(*b"iinf", 0)?;
        b.u16(self.items.len() as _)?;
        for infe in self.items.iter() {
            infe.write(&mut b)?;
        }
        Ok(())
    }
}

/// Item Info Entry box
#[derive(Debug, Copy, Clone)]
pub struct InfeBox {
    pub id: u16,
    pub typ: FourCC,
    pub name: &'static str,
}

impl MpegBox for InfeBox {
    #[inline(always)]
    fn len(&self) -> usize {
        FULL_BOX_SIZE
        + 2 // id
        + 2 // item_protection_index
        + 4 // type
        + self.name.as_bytes().len() + 1 // nul-terminated
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.full_box(*b"infe", 2)?;
        b.u16(self.id)?;
        b.u16(0)?;
        b.push(&self.typ.0)?;
        b.push(self.name.as_bytes())?;
        b.u8(0)
    }
}

#[derive(Debug, Clone)]
pub struct HdlrBox {
}

impl MpegBox for HdlrBox {
    #[inline(always)]
    fn len(&self) -> usize {
        FULL_BOX_SIZE + 4 + 4 + 13
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        // because an image format needs to be told it's an image format,
        // and it does it the way classic MacOS used to, because Quicktime.
        b.full_box(*b"hdlr", 0)?;
        b.u32(0)?; // old MacOS file type handler
        b.push(b"pict")?; // MacOS Quicktime subtype
        b.u32(0)?; // Firefox 92 wants all 0 here
        b.u32(0)?; // Reserved
        b.u32(0)?; // Reserved
        b.u8(0)?; // Pascal string for component name
        Ok(())
    }
}

/// Item properties + associations
#[derive(Debug, Clone)]
pub struct IprpBox {
    pub ipco: IpcoBox,
    pub ipma: IpmaBox,
}

impl MpegBox for IprpBox {
    #[inline(always)]
    fn len(&self) -> usize {
        BASIC_BOX_SIZE
            + self.ipco.len()
            + self.ipma.len()
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.basic_box(*b"iprp")?;
        self.ipco.write(&mut b)?;
        self.ipma.write(&mut b)
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum IpcoProp {
    Av1C(Av1CBox),
    Pixi(PixiBox),
    Ispe(IspeBox),
    AuxC(AuxCBox),
    Colr(ColrBox),
}

impl IpcoProp {
    pub fn len(&self) -> usize {
        match self {
            Self::Av1C(p) => p.len(),
            Self::Pixi(p) => p.len(),
            Self::Ispe(p) => p.len(),
            Self::AuxC(p) => p.len(),
            Self::Colr(p) => p.len(),
        }
    }

    pub fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        match self {
            Self::Av1C(p) => p.write(w),
            Self::Pixi(p) => p.write(w),
            Self::Ispe(p) => p.write(w),
            Self::AuxC(p) => p.write(w),
            Self::Colr(p) => p.write(w),
        }
    }
}

/// Item Property Container box
#[derive(Debug, Clone)]
pub struct IpcoBox {
    props: ArrayVec<IpcoProp, 7>,
}

impl IpcoBox {
    pub fn new() -> Self {
        Self { props: ArrayVec::new() }
    }

    pub fn push(&mut self, prop: IpcoProp) -> u8 {
        self.props.push(prop);
        self.props.len() as u8 // the spec wants them off by one
    }
}

impl MpegBox for IpcoBox {
    #[inline]
    fn len(&self) -> usize {
        BASIC_BOX_SIZE
            + self.props.iter().map(|a| a.len()).sum::<usize>()
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.basic_box(*b"ipco")?;
        for p in self.props.iter() {
            p.write(&mut b)?;
        }
        Ok(())
    }
}

#[derive(Debug, Copy, Clone)]
pub struct AuxCBox {
    pub urn: &'static str,
}

impl AuxCBox {
    pub fn len(&self) -> usize {
        FULL_BOX_SIZE + self.urn.len() + 1
    }

    pub fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.full_box(*b"auxC", 0)?;
        b.push(self.urn.as_bytes())?;
        b.u8(0)
    }
}

/// Pixies, I guess.
#[derive(Debug, Copy, Clone)]
pub struct PixiBox {
    pub depth: u8,
    pub channels: u8,
}

impl PixiBox {
    pub fn len(&self) -> usize {
        FULL_BOX_SIZE
            + 1 + self.channels as usize
    }

    pub fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.full_box(*b"pixi", 0)?;
        b.u8(self.channels)?;
        for _ in 0..self.channels {
            b.u8(self.depth)?;
        }
        Ok(())
    }
}

/// This is HEVC-specific and not for AVIF, but Chrome wants it :(
#[derive(Debug, Copy, Clone)]
pub struct IspeBox {
    pub width: u32,
    pub height: u32,
}

impl MpegBox for IspeBox {
    #[inline(always)]
    fn len(&self) -> usize {
        FULL_BOX_SIZE + 4 + 4
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.full_box(*b"ispe", 0)?;
        b.u32(self.width)?;
        b.u32(self.height)
    }
}

/// Property→image associations
#[derive(Debug, Clone)]
pub struct IpmaEntry {
    pub item_id: u16,
    pub prop_ids: ArrayVec<u8, 5>,
}

#[derive(Debug, Clone)]
pub struct IpmaBox {
    pub entries: ArrayVec<IpmaEntry, 2>,
}

impl MpegBox for IpmaBox {
    #[inline]
    fn len(&self) -> usize {
        FULL_BOX_SIZE + 4 + self.entries.iter().map(|e| 2 + 1 + e.prop_ids.len()).sum::<usize>()
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.full_box(*b"ipma", 0)?;
        b.u32(self.entries.len() as _)?; // entry count

        for e in &self.entries {
            b.u16(e.item_id)?;
            b.u8(e.prop_ids.len() as u8)?; // assoc count
            for &p in e.prop_ids.iter() {
                b.u8(p)?;
            }
        }
        Ok(())
    }
}

/// Item Reference box
#[derive(Debug, Copy, Clone)]
pub struct IrefEntryBox {
    pub from_id: u16,
    pub to_id: u16,
    pub typ: FourCC,
}

impl MpegBox for IrefEntryBox {
    #[inline(always)]
    fn len(&self) -> usize {
        BASIC_BOX_SIZE
            + 2 // from
            + 2 // refcount
            + 2 // to
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.basic_box(self.typ.0)?;
        b.u16(self.from_id)?;
        b.u16(1)?;
        b.u16(self.to_id)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub struct IrefBox {
    pub entry: IrefEntryBox,
}

impl MpegBox for IrefBox {
    #[inline(always)]
    fn len(&self) -> usize {
        FULL_BOX_SIZE + self.entry.len()
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.full_box(*b"iref", 0)?;
        self.entry.write(&mut b)
    }
}

#[derive(Debug, Clone)]
struct IrefBox2 {
    pub entries: ArrayVec<IrefEntryBox, 2>,
}

impl MpegBox for IrefBox2 {
    #[inline(always)]
    fn len(&self) -> usize {
        FULL_BOX_SIZE + self.entries.iter().map(|e| e.len()).sum::<usize>()
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.full_box(*b"iref", 0)?;
        for entry in &self.entries {
            entry.write(&mut b)?
        }
        Ok(())
    }
}

/// Auxiliary item (alpha or depth map)
#[derive(Debug, Copy, Clone)]
pub struct AuxlBox {}

impl MpegBox for AuxlBox {
    #[inline(always)]
    fn len(&self) -> usize {
        FULL_BOX_SIZE
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.full_box(*b"auxl", 0)
    }
}

/// ColourInformationBox
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ColrBox {
    pub color_primaries: ColorPrimaries,
    pub transfer_characteristics: TransferCharacteristics,
    pub matrix_coefficients: MatrixCoefficients,
    pub full_range_flag: bool, // u1 + u7
}

impl Default for ColrBox {
    fn default() -> Self {
        Self {
            color_primaries: ColorPrimaries::Bt709,
            transfer_characteristics: TransferCharacteristics::Srgb,
            matrix_coefficients: MatrixCoefficients::Bt601,
            full_range_flag: true,
        }
    }
}

impl MpegBox for ColrBox {
    #[inline(always)]
    fn len(&self) -> usize {
        BASIC_BOX_SIZE + 4 + 2 + 2 + 2 + 1
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.basic_box(*b"colr")?;
        b.u32(u32::from_be_bytes(*b"nclx"))?;
        b.u16(self.color_primaries as u16)?;
        b.u16(self.transfer_characteristics as u16)?;
        b.u16(self.matrix_coefficients as u16)?;
        b.u8(if self.full_range_flag { 1 << 7 } else { 0 })
    }
}
#[derive(Debug, Copy, Clone)]
pub struct Av1CBox {
    pub seq_profile: u8,
    pub seq_level_idx_0: u8,
    pub seq_tier_0: bool,
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub monochrome: bool,
    pub chroma_subsampling_x: bool,
    pub chroma_subsampling_y: bool,
    pub chroma_sample_position: u8,
}

impl MpegBox for Av1CBox {
    #[inline(always)]
    fn len(&self) -> usize {
        BASIC_BOX_SIZE + 4
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.basic_box(*b"av1C")?;
        let flags1 =
            u8::from(self.seq_tier_0) << 7 |
            u8::from(self.high_bitdepth) << 6 |
            u8::from(self.twelve_bit) << 5 |
            u8::from(self.monochrome) << 4 |
            u8::from(self.chroma_subsampling_x) << 3 |
            u8::from(self.chroma_subsampling_y) << 2 |
            self.chroma_sample_position;

        b.push(&[
            0x81, // marker and version
            (self.seq_profile << 5) | self.seq_level_idx_0, // x2d == 45
            flags1,
            0,
        ])
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PitmBox(pub u16);

impl MpegBox for PitmBox {
    #[inline(always)]
    fn len(&self) -> usize {
        FULL_BOX_SIZE + 2
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.full_box(*b"pitm", 0)?;
        b.u16(self.0)
    }
}

#[derive(Debug, Clone)]
pub struct IlocBox {
    pub items: ArrayVec<IlocItem, 2>,
}

#[derive(Debug, Clone)]
pub struct IlocItem {
    pub id: u16,
    pub extents: ArrayVec<IlocExtent, 1>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum IlocOffset {
    Relative(usize),
    Absolute(u32),
}

#[derive(Debug, Copy, Clone)]
pub struct IlocExtent {
    pub offset: IlocOffset,
    pub len: usize,
}

impl MpegBox for IlocBox {
    #[inline(always)]
    fn len(&self) -> usize {
        FULL_BOX_SIZE
        + 1 // offset_size, length_size
        + 1 // base_offset_size, reserved
        + 2 // num items
        + self.items.iter().map(|i| ( // for each item
            2 // id
            + 2 // dat ref idx
            + 0 // base_offset_size
            + 2 // extent count
            + i.extents.len() * ( // for each extent
               4 // extent_offset
               + 4 // extent_len
            )
        )).sum::<usize>()
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.full_box(*b"iloc", 0)?;
        b.push(&[4 << 4 | 4, 0])?; // offset and length are 4 bytes

        b.u16(self.items.len() as _)?; // num items
        for item in self.items.iter() {
            b.u16(item.id)?;
            b.u16(0)?;
            b.u16(item.extents.len() as _)?; // num extents
            for ex in &item.extents {
                b.u32(match ex.offset {
                    IlocOffset::Absolute(val) => val,
                    IlocOffset::Relative(_) => panic!("absolute offset must be set"),
                })?;
                b.u32(ex.len as _)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MdatBox<'data> {
    pub data_chunks: ArrayVec<&'data [u8], 4>,
}

impl MpegBox for MdatBox<'_> {
    #[inline(always)]
    fn len(&self) -> usize {
        BASIC_BOX_SIZE + self.data_chunks.iter().map(|c| c.len()).sum::<usize>()
    }

    fn write<B: WriterBackend>(&self, w: &mut Writer<B>) -> Result<(), B::Error> {
        let mut b = w.new_box(self.len());
        b.basic_box(*b"mdat")?;
        for ch in &self.data_chunks {
            b.push(ch)?;
        }
        Ok(())
    }
}
