use once_cell::sync::Lazy;
use pyo3::prelude::*;
use rand::{rng, Rng};
use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;

use crate::action::Action;
use crate::heading::Heading;
use crate::state::{extract_state, State};

// RL formula Q(s,a)←Q(s,a)+α[r+γa′max​Q(s′,a′)−Q(s,a)]
#[derive(Debug, Clone)]
pub struct QAgent {
    alpha: f32,                        // Learning rate
    gamma: f32, // Discount factor γ (small numbers prioritize immediate rewards, while bigger numbers encourage long-term rewards)
    epsilon: f32, // Exploration rate ε
    epsilon_min: f32, // Minimum exploration rate ε_min
    epsilon_decay: f32, // Decay rate for exploration ε_decay
    q_table: HashMap<State, [f32; 3]>, // Forward, Left, Right
}

pub static AGENT: Lazy<Mutex<Option<QAgent>>> = Lazy::new(|| Mutex::new(None));

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

    pub fn update(
        &mut self,
        old_state: State,
        action: Action,
        reward: f32,
        next_state: Option<State>,
    ) {
        let a_idx = action as usize;

        let target = if let Some(s2) = next_state {
            let q_next = self.q_table.entry(s2).or_insert([0.0; 3]);
            let max_next = q_next.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            reward + self.gamma * max_next
        } else {
            reward
        };

        let q_s = self.q_table.entry(old_state).or_insert([0.0; 3]);
        let old = q_s[a_idx];
        q_s[a_idx] = old + self.alpha * (target - old);

        println!(
            "Updated Q-value for state {:?}, action {:?}: {:.3} -> {:.3}",
            old_state, action, old, q_s[a_idx]
        );
    }

    pub fn decay(&mut self) {
        // Epsilon decay means it'd start to explore less and exploit more over time
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    pub fn learn(
        prev_view: Vec<Vec<String>>,
        prev_heading: (i32, i32),
        action: (i32, i32),
        reward: f32,
        // if done, there will be no next_view or heading
        next_view: Option<Vec<Vec<String>>>,
        next_heading: Option<(i32, i32)>,
        done: bool,
    ) -> PyResult<()> {
        let mut agent_guard = AGENT.lock().unwrap();
        let agent = agent_guard.as_mut().expect("Agent not initialized.");

        let prev_h = Heading::from_tuple(prev_heading.0, prev_heading.1).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid prev heading")
        })?;
        let s = extract_state(&prev_view, prev_h);

        // figure out which relative action corresponds to the absolute action applied
        let absolute_a = Heading::from_tuple(action.0, action.1).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid action tuple")
        })?;
        let relative_action = Action::from_absolute(prev_h, absolute_a);

        let s_next: Option<State> = if done {
            None
        } else {
            let nh = next_heading
                .and_then(|(dx, dy)| Heading::from_tuple(dx, dy))
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid next heading")
                })?;
            Some(extract_state(next_view.as_ref().unwrap(), nh))
        };

        agent.update(s, relative_action, reward, s_next);
        agent.decay();
        Ok(())
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
