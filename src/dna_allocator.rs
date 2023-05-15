use std::marker::PhantomData;

pub trait CloneReallocationMemoryBuffer<Dna>
{
    fn clone_from_dna(&mut self, other_dna: &Dna) -> Dna;
    fn deallocate(&mut self, dna: Dna);
}

#[derive(Clone)]
pub struct SimpleCloneAllocator<T: Clone>
{
    pub(crate) phantom: PhantomData<T>
}

impl<T: Clone> CloneReallocationMemoryBuffer<T> for SimpleCloneAllocator<T>
{
    fn clone_from_dna(&mut self, other_dna: &T) -> T {
        other_dna.clone()
    }

    fn deallocate(&mut self, _dna: T) {
    }
}
