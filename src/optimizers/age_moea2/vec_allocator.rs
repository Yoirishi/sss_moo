use crate::buffer_allocator::{Allocator};

pub struct VecAllocator
{
    default_vector_capacity: usize
}

impl VecAllocator {
    pub fn new(default_vector_capacity: usize) -> Self 
    {
        VecAllocator
        {
            default_vector_capacity
        }
    }
}

impl<T> Allocator<Vec<T>> for VecAllocator
{
    fn alloc(&mut self) -> Vec<T> {
        Vec::with_capacity(self.default_vector_capacity)
    }
}

