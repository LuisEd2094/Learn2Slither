#[derive(Copy, Clone, Debug)]
pub enum Action {
    Forward = 0,
    Left = 1,
    Right = 2,
}

impl Action {
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Action::Forward,
            1 => Action::Left,
            2 => Action::Right,
            _ => panic!("Invalid action index"),
        }
    }
}
