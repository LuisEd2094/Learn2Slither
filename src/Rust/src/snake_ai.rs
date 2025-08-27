use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::sync::Mutex;

use crate::heading::Heading;
use crate::qagent::QAgent;
use crate::state::extract_state;

static AGENT: Lazy<Mutex<Option<QAgent>>> = Lazy::new(|| Mutex::new(None));

#[pyfunction]
pub fn act(snake_view: Vec<Vec<String>>, heading: (i32, i32)) -> PyResult<(i32, i32)> {
    let mut guard = AGENT.lock().unwrap();
    let agent = guard
        .as_mut()
        .expect("Agent not initialized. Call init(...) first.");

    let heading = Heading::from_tuple(heading.0, heading.1)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid heading tuple"))?;

    let s = extract_state(&snake_view, heading);

    // choose action
    let a = agent.eps_greedy(s);

    // map to new absolute heading & return (dx,dy)
    let new_heading = heading.turn(a);
    Ok(new_heading.to_tuple())
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

    Ok(())
}
