use crate::{
    server::{Binding, ComputeServer},
    storage_id_type,
};

// This ID is used to map a handle to its actual data.
storage_id_type!(StorageId);

/// Defines if data uses a full memory chunk or a slice of it.
#[derive(Clone, Debug)]
pub struct StorageUtilization {
    /// The offset in bytes from the chunk start.
    pub offset: u64,
    /// The size of the slice in bytes.
    pub size: u64,
}

/// Contains the [storage id](StorageId) of a resource and the way it is used.
#[derive(new, Clone, Debug)]
pub struct StorageHandle {
    /// Storage id.
    pub id: StorageId,
    /// How the storage is used.
    pub utilization: StorageUtilization,
}

impl StorageHandle {
    /// Returns the size the handle is pointing to in memory.
    pub fn size(&self) -> u64 {
        self.utilization.size
    }

    /// Returns the size the handle is pointing to in memory.
    pub fn offset(&self) -> u64 {
        self.utilization.offset
    }

    /// Increase the current offset with the given value in bytes.
    pub fn offset_start(&self, offset_bytes: u64) -> Self {
        let utilization = StorageUtilization {
            offset: self.offset() + offset_bytes,
            size: self.size() - offset_bytes,
        };

        Self {
            id: self.id,
            utilization,
        }
    }

    /// Reduce the size of the memory handle..
    pub fn offset_end(&self, offset_bytes: u64) -> Self {
        let utilization = StorageUtilization {
            offset: self.offset(),
            size: self.size() - offset_bytes,
        };

        Self {
            id: self.id,
            utilization,
        }
    }
}

/// Storage types are responsible for allocating and deallocating memory.
pub trait ComputeStorage: Send {
    /// The resource associated type determines the way data is implemented and how
    /// it can be accessed by kernels.
    type Resource: Send;

    /// The alignment memory is allocated with in this storage.
    const ALIGNMENT: u64;

    /// Returns the underlying resource for a specified storage handle
    fn get(&mut self, handle: &StorageHandle) -> Self::Resource;

    /// Allocates `size` units of memory and returns a handle to it
    fn alloc(&mut self, size: u64) -> StorageHandle;

    /// Deallocates the memory pointed by the given storage id.
    fn dealloc(&mut self, id: StorageId);
}

/// Access to the underlying resource for a given binding.
#[derive(new)]
pub struct BindingResource<Server: ComputeServer> {
    // This binding is here just to keep the underlying allocation alive.
    // If the underlying allocation becomes invalid, someone else might
    // allocate into this resource which could lead to bad behaviour.
    #[allow(unused)]
    binding: Binding,
    resource: <Server::Storage as ComputeStorage>::Resource,
}

impl<Server: ComputeServer> BindingResource<Server> {
    /// access the underlying resource. Note: The resource might be bigger
    /// than just the original allocation for the binding. Only the part
    /// for the original binding is guaranteed to remain, other parts
    /// of the resource *will* be re-used.
    pub fn resource(&self) -> &<Server::Storage as ComputeStorage>::Resource {
        &self.resource
    }
}
