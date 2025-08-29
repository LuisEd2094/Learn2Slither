use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::action::Action;
use crate::heading::Heading;
use crate::qagent::QAgent;
use crate::qagent::AGENT;
use crate::state::{extract_state, State};

#[pyfunction]
pub fn act(state: State, heading: (i32, i32)) -> PyResult<(i32, i32)> {
    let mut guard = AGENT.lock().unwrap();
    let agent = guard
        .as_mut()
        .expect("Agent not initialized. Call init(...) first.");

    // choose action
    let heading = Heading::from_tuple(heading.0, heading.1)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid heading tuple"))?;
    let a = agent.eps_greedy(state);

    // map to new absolute heading & return (dx,dy)
    let new_heading = heading.turn(a);
    Ok(new_heading.to_tuple())
}

#[pyfunction]
pub fn get_q_table() -> PyResult<Vec<(u64, f32, f32, f32)>> {
    let mut guard = AGENT.lock().unwrap();
    let agent = guard
        .as_mut()
        .expect("Agent not initialized. Call init(...) first.");

    Ok(agent.get_q_table())
}

#[pyfunction]
pub fn load_q_table(data: Vec<(u64, f32, f32, f32)>) -> PyResult<()> {
    let mut guard = AGENT.lock().unwrap();
    let agent = guard
        .as_mut()
        .expect("Agent not initialized. Call init(...) first.");

    agent.load_q_table(data);
    Ok(())
}
/* use std::collections::HashMap;
pub fn print_q_table(q_table: &HashMap<State, [f32; 3]>) {
    for (state, values) in q_table {
        let code = state.0;

        // Binary repr
        println!("State {} (0b{:07b}):", code, code);

        // Decode apple bits
        println!(
            "  Apples: front={} left={} right={} back={}",
            (code >> 0) & 1,
            (code >> 1) & 1,
            (code >> 2) & 1,
            (code >> 3) & 1,
        );

        // Decode obstacle bits
        println!(
            "  Obstacles: front={} left={} right={}",
            (code >> 4) & 1,
            (code >> 5) & 1,
            (code >> 6) & 1,
        );

        // Q-values
        println!(
            "  Q-values => Forward: {:.3}, Left: {:.3}, Right: {:.3}",
            values[0], values[1], values[2]
        );
        println!("---------------------------------------");
    }
} */

#[pyfunction]
pub fn get_numstates() -> PyResult<(i32, i32)> {
    let agent_guard = AGENT.lock().unwrap();
    if let Some(agent) = &*agent_guard {
        //print_q_table(&agent.q_table);
        let num_states = agent.q_table.len() as i32;

        // also count total state-action entries (each state has up to 3 Q-values)
        let total_entries: i32 = agent.q_table.values().map(|arr| arr.len() as i32).sum();

        Ok((num_states, total_entries))
    } else {
        Ok((0, 0))
    }
}

#[pyfunction]
pub fn get_action(prev_h: (i32, i32), new_h: (i32, i32)) -> PyResult<Action> {
    let prev_h: Heading = Heading::from_tuple(prev_h.0, prev_h.1)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid prev heading"))?;

    let new_h: Heading = Heading::from_tuple(new_h.0, new_h.1)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid new heading"))?;

    let action = Action::from_absolute(prev_h, new_h);
    Ok(action)
}

#[pyfunction]
pub fn get_state(
    snake_view: Vec<Vec<String>>,
    heading: (i32, i32),
    snake_head: (i32, i32),
) -> PyResult<State> {
    let heading = Heading::from_tuple(heading.0, heading.1)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid heading tuple"))?;
    let state = extract_state(&snake_view, heading, snake_head);
    Ok(state)
}

#[pyfunction]
pub fn learn(
    prev_state: State,
    action: Action,
    reward: f32,
    decay: bool,
    // If game over, there is no new view
    s_next: Option<State>,
) -> PyResult<()> {
    let mut agent_guard = AGENT.lock().unwrap();
    let agent = agent_guard.as_mut().expect("Agent not initialized.");

    agent.update(prev_state, action, reward, s_next);
    if decay {
        agent.decay();
    }
    Ok(())
}

#[pyfunction]
pub fn init(alpha: f32, gamma: f32, epsilon: f32, epsilon_min: f32, epsilon_decay: f32) {
    let mut agent_guard = AGENT.lock().unwrap();

    if agent_guard.is_some() {
    } else {
        *agent_guard = Some(QAgent::new(
            alpha,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
        ));
    }
}

#[pymodule]
fn snake_ai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(act, m)?)?;
    m.add_function(wrap_pyfunction!(learn, m)?)?;
    m.add_function(wrap_pyfunction!(get_numstates, m)?)?;
    m.add_function(wrap_pyfunction!(get_state, m)?)?;
    m.add_function(wrap_pyfunction!(get_action, m)?)?;
    m.add_function(wrap_pyfunction!(get_q_table, m)?)?;
    m.add_function(wrap_pyfunction!(load_q_table, m)?)?;

    m.add_class::<State>()?;
    m.add_class::<Action>()?;

    Ok(())
}
