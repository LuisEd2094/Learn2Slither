use crate::Heading;

#[derive(Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub struct State(u64);

/// What the snake might see on a ray
#[derive(Copy, Clone, Debug)]
enum Obj {
    Empty = 0,
    Wall = 1,
    Snake = 2,
    Green = 3,
    Red = 4,
}

impl Obj {
    fn from_char(c: char) -> Option<Self> {
        match c {
            'W' => Some(Obj::Wall),
            'S' => Some(Obj::Snake),
            'G' => Some(Obj::Green),
            'R' => Some(Obj::Red),
            '.' => Some(Obj::Empty),
            ' ' => None,
            _ => None,
        }
    }
}

fn bucket_distance(d: Option<usize>) -> u8 {
    match d {
        None => 4,
        Some(1) => 0,
        Some(x) if x <= 3 => 1,
        Some(x) if x <= 6 => 2,
        Some(x) if x <= 10 => 3,
        Some(_) => 3,
    }
}

// helper to probe a ray
fn first_on_ray(
    view: &[Vec<String>],
    mut x: isize,
    mut y: isize,
    (dx, dy): (isize, isize),
) -> (Obj, u8) {
    let h = view.len() as isize;
    let w = if h > 0 { view[0].len() as isize } else { 0 };

    let mut steps: usize = 0;
    loop {
        x += dx;
        y += dy;
        if x < 0 || y < 0 || x >= w || y >= h {
            return (Obj::Wall, bucket_distance(Some(steps + 1)));
        }
        steps += 1;
        let ch = view[y as usize][x as usize].chars().next().unwrap_or(' ');
        if let Some(obj) = Obj::from_char(ch) {
            if let Obj::Empty = obj {
                continue;
            } else {
                return (obj, bucket_distance(Some(steps)));
            }
        } else {
            // space: outside the cross; keep going
            continue;
        }
    }
}

/// Encode 4 (obj, distBucket) pairs + heading into a compact u64
/// Order of directions (fixed): Up, Down, Left, Right
fn encode_state(rays: &[(Obj, u8); 4], heading: Heading) -> State {
    // Layout: [ heading(2 bits) ][ for each of 4 dirs: obj(3 bits), dist(3 bits) ] = 2 + 4*(3+3) = 26 bits
    let mut code: u64 = 0;
    let head_bits: u64 = match heading {
        Heading::Up => 0,
        Heading::Down => 1,
        Heading::Left => 2,
        Heading::Right => 3,
    };
    code |= head_bits;
    let mut shift = 2;
    for (obj, dist) in rays {
        let o = *obj as u64 & 0b111; // 3 bits
        let d = *dist as u64 & 0b111; // 3 bits
        code |= o << shift;
        shift += 3;
        code |= d << shift;
        shift += 3;
    }
    println!("Encoded state = {:026b} (raw: {})", code, code);
    State(code)
}

/// Extract rays (Up,Down,Left,Right) from the 2D view:
/// - Find the head 'S' (we assume the head is on the cross center; if multiple 'S', pick the one whose row & col show the cross)
/// - For each direction, walk outward until you hit the first meaningful symbol
pub fn extract_state(view: &[Vec<String>], heading: Heading) -> State {
    let h = view.len();
    let w = if h > 0 { view[0].len() } else { 0 };

    // locate head 'S'
    let mut head: Option<(usize, usize)> = None;
    'outer: for (y, row) in view.iter().enumerate().take(h) {
        for (x, item) in row.iter().enumerate().take(w) {
            if *item == "S" {
                head = Some((x, y));
                break 'outer;
            }
        }
    }
    let (hx, hy) = head.expect("No head 'S' found in snake_view");

    let up = first_on_ray(view, hx as isize, hy as isize, (0, -1));
    let down = first_on_ray(view, hx as isize, hy as isize, (0, 1));
    let left = first_on_ray(view, hx as isize, hy as isize, (-1, 0));
    let right = first_on_ray(view, hx as isize, hy as isize, (1, 0));

    let rays = [up.0, down.0, left.0, right.0]
        .iter()
        .zip([up.1, down.1, left.1, right.1].iter())
        .map(|(o, d)| (*o, *d))
        .collect::<Vec<_>>();

    let rays: [(Obj, u8); 4] = [rays[0], rays[1], rays[2], rays[3]];
    encode_state(&rays, heading)
}
