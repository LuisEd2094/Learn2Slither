use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[derive(Clone)]
enum AIDirection {
    Forward,
    Left,
    Right,
}

impl AIDirection {
    /// Convert to (dx, dy) tuple for Python
    fn to_tuple(&self) -> (i32, i32) {
        match self {
            AIDirection::Forward => (0, -1),
            AIDirection::Left => (-1, 0),
            AIDirection::Right => (1, 0),
        }
    }
}

#[pyfunction]
pub fn choose_direction(snake_view: Vec<Vec<String>>) -> PyResult<(i32, i32)> {
    // Pick a direction (for now always Forward)
    println!("Snake view: {:?}", snake_view);
    let _choice = AIDirection::Forward;
    let _choice = AIDirection::Left;
    let choice = AIDirection::Right;
    Ok(choice.to_tuple())
}

#[pymodule]
fn snake_ai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(choose_direction, m)?)?;
    Ok(())
}
