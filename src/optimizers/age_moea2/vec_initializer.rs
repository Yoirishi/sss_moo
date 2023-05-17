use crate::buffer_allocator::Initializer;

pub struct VecInitializer {}

impl<T> Initializer<Vec<T>> for VecInitializer
{
    fn init(&mut self, obj: &mut Vec<T>) -> () {
        obj.clear()
    }
}