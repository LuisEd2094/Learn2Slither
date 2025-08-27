use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::action::Action;
use crate::heading::Heading;
use crate::qagent::QAgent;
use crate::qagent::AGENT;
use crate::state::{extract_state, State};

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
pub fn learn(
    prev_view: Vec<Vec<String>>,
    prev_heading: (i32, i32),
    action: (i32, i32),
    reward: f32,
    // If game over, there is no new view
    next_view: Option<Vec<Vec<String>>>,
    next_heading: Option<(i32, i32)>,
    done: bool,
) -> PyResult<()> {
    let mut agent_guard = AGENT.lock().unwrap();
    let agent = agent_guard.as_mut().expect("Agent not initialized.");

    let prev_h: Heading = Heading::from_tuple(prev_heading.0, prev_heading.1)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid prev heading"))?;
    let s: State = extract_state(&prev_view, prev_h);

    // figure out which relative action corresponds to the absolute action applied
    let absolute_action: Heading = Heading::from_tuple(action.0, action.1)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid action tuple"))?;

    let relative_action: Action = Action::from_absolute(prev_h, absolute_action);
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

    Ok(())
}
