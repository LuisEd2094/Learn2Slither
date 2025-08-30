use crate::action::Action;

/// Cardinal directions for absolute movement
#[derive(Copy, Clone, Debug)]
pub enum Heading {
    Up,
    Down,
    Left,
    Right,
}

impl Heading {
    pub fn from_tuple(dx: i32, dy: i32) -> Option<Self> {
        match (dx, dy) {
            (0, -1) => Some(Heading::Up),
            (0, 1) => Some(Heading::Down),
            (-1, 0) => Some(Heading::Left),
            (1, 0) => Some(Heading::Right),
            _ => None,
        }
    }

    pub fn to_tuple(self) -> (i32, i32) {
        match self {
            Heading::Up => (0, -1),
            Heading::Down => (0, 1),
            Heading::Left => (-1, 0),
            Heading::Right => (1, 0),
        }
    }

    pub fn turn(self, a: Action) -> Heading {
        match (self, a) {
            (Heading::Up, Action::Forward) => Heading::Up,
            (Heading::Up, Action::Left) => Heading::Left,
            (Heading::Up, Action::Right) => Heading::Right,

            (Heading::Down, Action::Forward) => Heading::Down,
            (Heading::Down, Action::Left) => Heading::Right,
            (Heading::Down, Action::Right) => Heading::Left,

            (Heading::Left, Action::Forward) => Heading::Left,
            (Heading::Left, Action::Left) => Heading::Down,
            (Heading::Left, Action::Right) => Heading::Up,

            (Heading::Right, Action::Forward) => Heading::Right,
            (Heading::Right, Action::Left) => Heading::Up,
            (Heading::Right, Action::Right) => Heading::Down,
        }
    }
}
