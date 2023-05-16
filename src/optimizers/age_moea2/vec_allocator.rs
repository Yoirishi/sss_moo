

struct VecAllocator<T>
{
    default_vec_capacity: usize,
    buf: Vec<T>
}

impl<T> VecAllocator<T>
{
    pub fn new() -> Self
    {
        VecAllocator {
            default_vec_capacity: 0,
            buf: vec![]
        }
    }

    pub fn new_with_default_vec_capacity(default_vec_capacity: usize) -> Self
    {
        VecAllocator {
            default_vec_capacity,
            buf: vec![]
        }
    }

    pub fn allocate(&mut self) -> Vec<T>
    {
        match self.buf.pop()
        {
            None => {
                if self.default_vec_capacity != 0
                {
                    Vec::with_capacity(self.default_vec_capacity)
                }
                else
                {
                    vec![]
                }
            }
            Some(vec) => {
                vec
            }
        }
    }

    pub fn clone_vec(&mut self, other_vec: &Vec<T>) -> Vec<T>
    {
        let mut new_vec = self.allocate();

        new_vec.clone_from(other_vec);

        new_vec
    }

    pub fn deallocate(&mut self, vec: Vec<T>)
    {
        self.buf.push(vec);
    }
}
