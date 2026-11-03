// Rubik's cube model and beginner-method solver, ported line for line from the MATLAB code in
// https://github.com/kjjoe/rubiks (Movement/, GetInfo/, Algorithms/, Stages/).
//
// Like the MATLAB version, a cube is a 5x5x5 array stored column-major (index i + 5j + 25k),
// with the 54 stickers on its outer shell and colours 1-6. MATLAB indices are 1-based, so the
// solver stages below keep 1-based piece indices to stay easy to compare with the original.
// The 54-sticker vector (colours 0-5) is the representation DeepCube uses.

const at = (i, j, k) => i + 5 * j + 25 * k;

export function newCube() {
  const c = new Int8Array(125);
  for (let a = 1; a <= 3; a++) {
    for (let b = 1; b <= 3; b++) {
      c[at(a, b, 0)] = 1; // white  (U)
      c[at(a, b, 4)] = 2; // yellow (D)
      c[at(0, a, b)] = 3; // red    (B)
      c[at(4, a, b)] = 4; // orange (F)
      c[at(a, 0, b)] = 5; // blue   (L)
      c[at(a, 4, b)] = 6; // green  (R)
    }
  }
  return c;
}

// ---------------------------------------------------------------------------
// Moves: each rotates whole slices of the 5x5x5 array with MATLAB's rot90
// ---------------------------------------------------------------------------

// index of element (a, b) of the 5x5 slice taken at position v along dimension dim
function sliceIndex(dim, v, a, b) {
  if (dim === 0) return at(v, a, b);
  if (dim === 1) return at(a, v, b);
  return at(a, b, v);
}

// dst slice = rot90(src slice, k), k = 1 (counter-clockwise), 2 or 3
function rotSlice(src, dst, dim, v, k) {
  for (let a = 0; a < 5; a++) {
    for (let b = 0; b < 5; b++) {
      let sa, sb;
      if (k === 1) { sa = b; sb = 4 - a; }
      else if (k === 3) { sa = 4 - b; sb = a; }
      else { sa = 4 - a; sb = 4 - b; }
      dst[sliceIndex(dim, v, a, b)] = src[sliceIndex(dim, v, sa, sb)];
    }
  }
}

const ALL = [0, 1, 2, 3, 4];
// face: [dimension, slices turned, rot90 count without prime] (from Movement/*.m)
const TURNS = {
  U: [2, [0, 1], 3], D: [2, [3, 4], 1],
  L: [1, [0, 1], 1], R: [1, [3, 4], 3],
  F: [0, [3, 4], 1], B: [0, [0, 1], 3],
  X: [1, ALL, 3], Y: [2, ALL, 3], Z: [0, ALL, 1],
  M: [1, [2], 1], E: [2, [2], 1], S: [0, [2], 1],
};

export function turn(cube, face, prime = false) {
  const [dim, slices, rot] = TURNS[face];
  const k = prime ? 4 - rot : rot;
  const out = cube.slice();
  for (const v of slices) rotSlice(cube, out, dim, v, k);
  return out;
}

export const tokens = (algo) =>
  (Array.isArray(algo) ? algo : String(algo).split(/\s+/)).filter(Boolean);

export function doAlgorithm(cube, algo) {
  for (const m of tokens(algo)) {
    if (m[1] === "2") cube = turn(turn(cube, m[0]), m[0]);
    else cube = turn(cube, m[0], m[1] === "'");
  }
  return cube;
}

export function invAlgo(algo) {
  return tokens(algo).reverse()
    .map((m) => (m.length === 2 ? (m[1] === "'" ? m[0] : m) : m + "'"))
    .join(" ");
}

// ---------------------------------------------------------------------------
// 54-sticker representation and move permutations (new_state[s] = old_state[perm[s]])
// ---------------------------------------------------------------------------

const SOLVED5 = newCube();
export const STICKERS = [...SOLVED5.keys()].filter((i) => SOLVED5[i] !== 0); // 54 slots, column-major order

export const toStickers = (cube) => Uint8Array.from(STICKERS, (i) => cube[i] - 1);
export function fromStickers(state) {
  const c = new Int8Array(125);
  STICKERS.forEach((i, s) => { c[i] = state[s] + 1; });
  return c;
}
export const SOLVED = toStickers(SOLVED5);

const permCache = new Map();
export function movePerm(name) {
  if (!permCache.has(name)) {
    const lab = new Int8Array(125);
    STICKERS.forEach((i, s) => { lab[i] = s + 1; });
    const out = doAlgorithm(lab, name);
    permCache.set(name, Int16Array.from(STICKERS, (i) => out[i] - 1));
  }
  return permCache.get(name);
}

// the 18 face turns, in the same order as the Python DeepCube code
export const FACE_MOVES = [..."UDLRFB"].flatMap((f) => [f, f + "'", f + "2"]);

export function applyMove(state, name) {
  const p = movePerm(name);
  return Uint8Array.from(p, (j) => state[j]);
}
export const applyAlgorithm = (state, algo) => tokens(algo).reduce(applyMove, state);
export const isSolved = (state) => state.every((v, i) => v === SOLVED[i]);

// ---------------------------------------------------------------------------
// Piece locations (GetInfo/)
// ---------------------------------------------------------------------------

const HALF = [[0, 1], [3, 4]];

export function getLocations(c) {
  const corners = [];
  const sides = [];
  for (let n = 0; n < 8; n++) {
    const [ri, rj, rk] = [HALF[n & 1], HALF[(n >> 1) & 1], HALF[(n >> 2) & 1]];
    const row = [];
    for (const k of rk) for (const j of rj) for (const i of ri) if (c[at(i, j, k)]) row.push(c[at(i, j, k)]);
    corners.push(row);
  }
  for (let n = 0; n < 4; n++) {
    const [r1, r2] = [HALF[n & 1], HALF[(n >> 1) & 1]];
    const a = [], b = [], d = [];
    for (const j of r2) for (const i of r1) if (c[at(i, j, 2)]) a.push(c[at(i, j, 2)]);
    for (const k of r2) for (const i of r1) if (c[at(i, 2, k)]) b.push(c[at(i, 2, k)]);
    for (const k of r2) for (const j of r1) if (c[at(2, j, k)]) d.push(c[at(2, j, k)]);
    sides[n] = a;
    sides[n + 4] = b;
    sides[n + 8] = d;
  }
  return { corners, sides };
}

// 1-based row of `matrix` holding the same colours as `color` (any order); last row if none
function getRowLoc(color, matrix) {
  for (let r = 0; r < matrix.length; r++) {
    let s = 0;
    for (const col of color) for (const x of matrix[r]) if (x === col) s++;
    if (s === color.length) return r + 1;
  }
  return matrix.length;
}

const eq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

function getSideMoves(index) {
  return {
    1: ["B", "B'", "L", "L'"], 2: ["F", "F'", "L", "L'"], 3: ["B", "B'", "R", "R'"],
    4: ["F", "F'", "R", "R'"], 5: ["B", "B'", "U", "U'"], 6: ["F", "F'", "U", "U'"],
    7: ["B", "B'", "D", "D'"], 8: ["F", "F'", "D", "D'"], 9: ["L", "L'", "U", "U'"],
    10: ["R", "R'", "U", "U'"], 11: ["L", "L'", "D", "D'"], 12: ["R", "R'", "D", "D'"],
  }[index];
}

function checkTopCorner(cube, cornersSolve) {
  const { corners } = getLocations(cube);
  let match = 0;
  for (let i = 1; i <= 4; i++) if (getRowLoc(cornersSolve[i - 1], corners) === i) match++;
  return match;
}

// ---------------------------------------------------------------------------
// Breadth-first searches (Algorithms/sideBFS.m, Algorithms/topsideBFS.m)
// ---------------------------------------------------------------------------

// shortest move sequence (never turning U) that brings one edge to its slot
function sideBFS(cube, indexGoal, sidesSolve) {
  const colorGoal = sidesSolve[indexGoal - 1];
  const key = (col, idx) => col.join(",") + "|" + idx;
  const goalKey = key(colorGoal, indexGoal);
  const allowed = new Set();
  for (let i = 1; i <= 12; i++) {
    allowed.add(key(colorGoal, i));
    allowed.add(key([...colorGoal].reverse(), i));
  }
  const sides = getLocations(cube).sides;
  const rootIdx = getRowLoc(colorGoal, sides);
  const root = { col: sides[rootIdx - 1], idx: rootIdx, cube, moves: [] };
  const seen = new Set();
  if (allowed.has(key(root.col, root.idx))) seen.add(key(root.col, root.idx));

  const queue = [root];
  let output;
  for (let q = 0; q < queue.length; q++) {
    const v = queue[q];
    if (key(v.col, v.idx) === goalKey) { output = v; continue; }
    for (const m of getSideMoves(v.idx)) {
      if (m[0] === "U") continue;
      const c2 = doAlgorithm(v.cube, m);
      const s2 = getLocations(c2).sides;
      const i2 = getRowLoc(colorGoal, s2);
      const k2 = key(s2[i2 - 1], i2);
      if (allowed.has(k2) && !seen.has(k2)) {
        seen.add(k2);
        queue.push({ col: s2[i2 - 1], idx: i2, cube: c2, moves: [...v.moves, m] });
      }
    }
  }
  return { cube: output.cube, algo: output.moves };
}

const SIDE_R_D = "L F R' F' L' R U R U' R'";
const SIDE_L_D = "R' F' L F R L' U' L' U L";

// sequence of edge algorithms that solves the last three top-layer edges
function topsideBFS(cube, sidesSolve) {
  const moveTop = [
    SIDE_R_D, invAlgo(SIDE_R_D),
    `Y Y ${SIDE_L_D} Y Y`, `Y Y ${invAlgo(SIDE_L_D)} Y Y`,
  ];
  const pick = (s) => [s[4], s[5], s[9]];
  const key = (st) => st.map((r) => r.join(",")).join(";");
  const goal = pick(sidesSolve);
  const goalKey = key(goal);
  const allowed = new Set();
  const orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
  for (let f = 0; f < 8; f++) {
    for (const p of orders) {
      allowed.add(key(p.map((pi, j) => ((f >> j) & 1 ? [...goal[pi]].reverse() : goal[pi]))));
    }
  }
  const root = { st: pick(getLocations(cube).sides), cube, moves: [] };
  const seen = new Set();
  if (allowed.has(key(root.st))) seen.add(key(root.st));

  const queue = [root];
  let output;
  for (let q = 0; q < queue.length; q++) {
    const v = queue[q];
    if (key(v.st) === goalKey) { output = v; continue; }
    for (const algo of moveTop) {
      const c2 = doAlgorithm(v.cube, algo);
      const st = pick(getLocations(c2).sides);
      const k2 = key(st);
      if (allowed.has(k2) && !seen.has(k2)) {
        seen.add(k2);
        queue.push({ st, cube: c2, moves: [...v.moves, algo] });
      }
    }
  }
  return { cube: output.cube, algo: output.moves.flatMap(tokens) };
}

// ---------------------------------------------------------------------------
// Solver stages (Stages/*.m). `st` holds the cube, the solved cube in the same orientation,
// and the moves recorded so far.
// ---------------------------------------------------------------------------

function play(st, ...algos) {
  for (const a of algos) {
    st.cube = doAlgorithm(st.cube, a);
    st.algo.push(...tokens(a));
  }
}
function rotateBoth(st, m) {
  st.cube = doAlgorithm(st.cube, m);
  st.solved = doAlgorithm(st.solved, m);
  st.algo.push(m);
}

// stage1p1 calls rng(1) and then randperm(4) once per pass; these are MATLAB's outputs
const RANDPERM = [
  [3, 4, 1, 2], [2, 1, 3, 4], [1, 3, 2, 4], [3, 1, 4, 2], [3, 4, 1, 2],
  [3, 4, 1, 2], [4, 3, 1, 2], [3, 1, 4, 2], [4, 2, 3, 1], [3, 1, 4, 2],
];

// stage 1.1: the cross
function stage1p1(st) {
  const sidesSolve = getLocations(st.solved).sides;
  let cross = [5, 6, 9, 10];
  let exitCross = false;
  let steps = 0;
  while (!exitCross && steps < 10) {
    let checkCross = 0;
    cross = RANDPERM[steps].map((p) => cross[p - 1]);
    for (const indexGoal of cross) {
      const r = sideBFS(st.cube, indexGoal, sidesSolve);
      st.cube = r.cube;
      if (r.algo.length === 0) checkCross++;
      else st.algo.push(...r.algo);
    }
    if (checkCross === 4) exitCross = true;
    steps++;
  }
}

// insert the corner in slot 4 by repeating R' D' R D until it is oriented
function orientCorner(st, colorGoal) {
  let correct = false;
  for (let steps = 0; !correct && steps < 6; steps++) {
    const { corners } = getLocations(st.cube);
    if (eq(corners[getRowLoc(colorGoal, corners) - 1], colorGoal)) correct = true;
    else play(st, "R' D' R D");
  }
}

// stage 1.2: first-layer corners, then flip the cube upside down
function stage1p2(st) {
  for (let y = 0; y < 4; y++) {
    rotateBoth(st, "Y");
    const colorGoal = getLocations(st.solved).corners[3];
    const indexGoal = 4;

    let indexRoot = getRowLoc(colorGoal, getLocations(st.cube).corners);
    if (indexRoot < 5 && indexRoot !== indexGoal) {
      play(st, ["L' D' L", "L D L'", "B' D' B", "R' D' R"][Math.min(indexRoot, 4) - 1]);
    }

    indexRoot = getRowLoc(colorGoal, getLocations(st.cube).corners);
    if (indexRoot !== indexGoal + 4 && indexRoot > 4) {
      let tmp = st.cube;
      let lowerPos = false;
      for (let steps = 0; !lowerPos && steps < 3; steps++) {
        tmp = turn(tmp, "D");
        st.algo.push("D");
        if (getRowLoc(colorGoal, getLocations(tmp).corners) === indexGoal + 4) {
          st.cube = tmp;
          lowerPos = true;
        }
      }
    }
    orientCorner(st, colorGoal);
  }
  rotateBoth(st, "Z2");
}

const MID_RIGHT = "U R U' R' U' F' U F";
const MID_LEFT = "U' L' U L U F U' F'";

// stage 2: middle-layer edges
function stage2(st) {
  for (let y = 0; y < 4; y++) {
    rotateBoth(st, "Y");
    const colorGoal = getLocations(st.solved).sides[3];
    let sides = getLocations(st.cube).sides;
    let idx = getRowLoc(colorGoal, sides);
    let same = eq(sides[idx - 1], colorGoal);
    const recheck = () => {
      sides = getLocations(st.cube).sides;
      return eq(sides[getRowLoc(colorGoal, sides) - 1], colorGoal);
    };

    if (idx === 4 && same) {
      // already solved
    } else if (idx === 4) {
      play(st, MID_RIGHT, "U2", MID_RIGHT);
    } else if (idx === 2) {
      play(st, MID_LEFT);
      if (recheck()) play(st, "U Y", MID_LEFT, "Y'");
      else play(st, "U2", MID_RIGHT);
    } else if (idx === 3) {
      play(st, "Y", MID_RIGHT, "Y'");
      if (recheck()) play(st, "U2 Y", MID_LEFT, "Y'");
      else play(st, "U'", MID_RIGHT);
    } else if (idx === 1) {
      play(st, "Y2", MID_RIGHT, "Y2");
      if (recheck()) play(st, "U' Y", MID_LEFT, "Y'");
      else play(st, MID_RIGHT);
    } else if (idx === 5) {
      if (same) play(st, "U Y", MID_LEFT, "Y'");
      else play(st, "U2", MID_RIGHT);
    } else if (idx === 6) {
      if (same) play(st, "U' Y", MID_LEFT, "Y'");
      else play(st, MID_RIGHT);
    } else if (idx === 9) {
      if (same) play(st, "U2 Y", MID_LEFT, "Y'");
      else play(st, "U'", MID_RIGHT);
    } else if (idx === 10) {
      if (same) play(st, "Y", MID_LEFT, "Y'");
      else play(st, "U", MID_RIGHT);
    }
  }
}

const ROT_CC = "U R U' L' U R' U' L";

// stage 3.1: position, then orient, the last-layer corners
function stage3p1(st) {
  let cornersSolve = getLocations(st.solved).corners;
  const colorGoal = cornersSolve[3];
  const idx = getRowLoc(colorGoal, getLocations(st.cube).corners);
  if (idx === 3) play(st, "U");
  else if (idx === 1) play(st, "U2");
  else if (idx === 2) play(st, "U'");

  let match = checkTopCorner(st.cube, cornersSolve);
  for (let steps = 0; match !== 4 && steps < 10; steps++) {
    if (match === 2) play(st, "U'", ROT_CC, "U2");
    else if (match === 1) play(st, ROT_CC);
    match = checkTopCorner(st.cube, cornersSolve);
  }

  for (let y = 0; y < 4; y++) {
    play(st, "U");
    st.solved = turn(st.solved, "U");
    cornersSolve = getLocations(st.solved).corners;
    orientCorner(st, cornersSolve[3]);
  }
}

// stage 3.2: last-layer edges
function stage3p2(st) {
  let sidesSolve = getLocations(st.solved).sides;
  const R_D = SIDE_R_D, R_U = invAlgo(SIDE_R_D), L_D = SIDE_L_D, L_U = invAlgo(SIDE_L_D);
  const colorGoal = sidesSolve[5];
  const sides = getLocations(st.cube).sides;
  const idx = getRowLoc(colorGoal, sides);
  const same = eq(colorGoal, sides[idx - 1]);
  if (idx === 9) {
    if (same) play(st, "Y2", R_U, "Y2");
    else play(st, L_D);
  } else if (idx === 5) {
    if (same) play(st, "Y2", L_U, "Y2", R_D);
    else play(st, R_U);
  } else if (idx === 10) {
    if (same) play(st, "Y2", L_U, "Y2");
    else play(st, R_D);
  } else if (idx === 6 && !same) {
    play(st, R_U, "Y2", L_U, "Y2");
  }
  rotateBoth(st, "Y");
  sidesSolve = getLocations(st.solved).sides;
  const r = topsideBFS(st.cube, sidesSolve);
  st.cube = r.cube;
  st.algo.push(...r.algo);
}

// Full beginner-method solve of a 5x5x5 cube. Returns the moves exactly as the MATLAB code
// records them, including whole-cube rotations (X, Y, Z).
export function beginnerSolve(cube) {
  const st = { cube, solved: newCube(), algo: [] };
  stage1p1(st);
  stage1p2(st);
  stage2(st);
  stage3p1(st);
  stage3p2(st);
  if (!eq(st.cube, st.solved)) throw new Error("beginner solver did not reach the solved state");
  return st.algo;
}

// ---------------------------------------------------------------------------
// Rewrite a solution with whole-cube rotations into face turns only, so it can be played on
// a cube that never changes orientation, then merge neighbouring turns of the same face.
// ---------------------------------------------------------------------------

export function toFaceTurns(algo) {
  let q = Int16Array.from({ length: 54 }, (_, i) => i); // rotation applied so far
  const out = [];
  for (const m of tokens(algo)) {
    const p = movePerm(m);
    if ("XYZ".includes(m[0])) {
      q = Int16Array.from(p, (pj) => q[pj]);
      continue;
    }
    // the face turn f that does the same thing in the cube's original frame
    const f = FACE_MOVES.find((name) => {
      const pf = movePerm(name);
      for (let j = 0; j < 54; j++) if (q[p[j]] !== pf[q[j]]) return false;
      return true;
    });
    out.push(f);
  }
  return simplify(out);
}

const AMOUNT = { "": 1, "'": 3, "2": 2 };
const SUFFIX = ["", "", "2", "'"];

export function simplify(moves) {
  const out = [];
  for (const m of moves) {
    const last = out[out.length - 1];
    if (last && last[0] === m[0]) {
      const n = (AMOUNT[last.slice(1)] + AMOUNT[m.slice(1)]) % 4;
      out.pop();
      if (n) out.push(m[0] + SUFFIX[n]);
    } else {
      out.push(m);
    }
  }
  return out;
}

// beginner-method solution for a 54-sticker state, as face turns
export function solveBeginner(state) {
  return toFaceTurns(beginnerSolve(fromStickers(state)));
}

export function randomScramble(n, rand = Math.random) {
  const moves = [];
  while (moves.length < n) {
    const m = FACE_MOVES[Math.floor(rand() * 18)];
    if (moves.length && moves[moves.length - 1][0] === m[0]) continue; // no back-to-back same face
    moves.push(m);
  }
  return moves;
}
