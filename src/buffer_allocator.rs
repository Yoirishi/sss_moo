
pub trait Allocator<T>
{
    fn alloc(&mut self) -> T;
}

pub struct BufferAllocator<T: Clone, AllocatorT: Allocator<T>>
{
    allocator: AllocatorT,
    buf: Vec<T>
}

impl<T: Clone, AllocatorT: Allocator<T>> BufferAllocator<T, AllocatorT>
{
    pub fn new(allocator: AllocatorT) -> Self
    {
        BufferAllocator {
            allocator,
            buf: vec![]
        }
    }

    pub fn allocate(&mut self) -> T
    {
        match self.buf.pop()
        {
            None => {
                self.allocator.alloc()
            }
            Some(vec) => {
                vec
            }
        }
    }

    pub fn clone_vec(&mut self, from_obj: &T) -> T
    {
        let mut new_obj = self.allocate();

        new_obj.clone_from(from_obj);

        new_obj
    }

    pub fn deallocate(&mut self, obj: T)
    {
        self.buf.push(obj);
    }
}
