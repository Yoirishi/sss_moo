
pub trait Allocator<T>
{
    fn alloc(&mut self) -> T;
}

pub trait Initializer<T>
{
    fn init(&mut self, obj: &mut T) -> ();
}

pub struct BufferAllocator<T: Clone, AllocatorT: Allocator<T>, InitializerT: Initializer<T>>
{
    initializer: InitializerT,
    allocator: AllocatorT,
    buf: Vec<T>
}

impl<T: Clone, 
    AllocatorT: Allocator<T>, 
    InitializerT: Initializer<T>> BufferAllocator<T, 
    AllocatorT, InitializerT>
{
    pub fn new(allocator: AllocatorT, initializer: InitializerT) -> Self
    {
        BufferAllocator {
            initializer,
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
            Some(mut vec) => {
                self.initializer.init(&mut vec);
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
