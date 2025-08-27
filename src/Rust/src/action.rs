use crate::heading::Heading;

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
    pub fn from_absolute(prev_h: Heading, new_h: Heading) -> Self {
        if new_h.to_tuple() == prev_h.to_tuple() {
            Action::Forward
        } else if new_h.to_tuple() == prev_h.turn(Action::Left).to_tuple() {
            Action::Left
        } else if new_h.to_tuple() == prev_h.turn(Action::Right).to_tuple() {
            Action::Right
        } else {
            // fallback (this shouldn't normally happen unless you allow 180° turns)
            panic!("Invalid action transition");
        }
    }
}
