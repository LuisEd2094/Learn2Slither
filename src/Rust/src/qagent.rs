use rand::rng;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;

use crate::action::Action;
use crate::state::State;

// RL formula Q(s,a)←Q(s,a)+α[r+γa′max​Q(s′,a′)−Q(s,a)]
#[derive(Debug, Clone)]
pub struct QAgent {
    alpha: f32,                        // Learning rate α
    gamma: f32,                        // Discount factor γ
    epsilon: f32,                      // Exploration rate ε
    epsilon_min: f32,                  // Minimum exploration rate ε_min
    epsilon_decay: f32,                // Decay rate for exploration ε_decay
    q_table: HashMap<State, [f32; 3]>, // Forward, Left, Right
}

impl QAgent {
    pub fn new(alpha: f32, gamma: f32, epsilon: f32, epsilon_min: f32, epsilon_decay: f32) -> Self {
        Self {
            alpha,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
            q_table: HashMap::new(),
        }
    }

    pub fn eps_greedy(&mut self, s: State) -> Action {
        // ensure entry exists
        let entry = self.q_table.entry(s).or_insert([0.0; 3]);
        let mut rng = rng();

        // exploration vs exploitation
        if rng.random::<f32>() < self.epsilon {
            println!("Exploring...");
            let idx = rng.random_range(0..3);
            Action::from_index(idx)
        } else {
            println!("Exploiting...");
            let (idx, _) = entry
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            Action::from_index(idx)
        }
    }
}

// Implement Display for QAgent
impl fmt::Display for QAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "QAgent:")?;
        writeln!(f, "  alpha: {:.3}", self.alpha)?;
        writeln!(f, "  gamma: {:.3}", self.gamma)?;
        writeln!(f, "  epsilon: {:.3}", self.epsilon)?;
        writeln!(f, "  epsilon_min: {:.3}", self.epsilon_min)?;
        writeln!(f, "  epsilon_decay: {:.3}", self.epsilon_decay)?;
        writeln!(f, "  Q-table entries: {}", self.q_table.len())?;

        // Print first few entries
        let mut count = 0;
        for (state, actions) in self.q_table.iter() {
            writeln!(
                f,
                "    State {:?} => [Forward: {:.3}, Left: {:.3}, Right: {:.3}]",
                state, actions[0], actions[1], actions[2]
            )?;
            count += 1;
            if count >= 5 {
                writeln!(f, "    ...")?;
                break;
            }
        }
        Ok(())
    }
}
