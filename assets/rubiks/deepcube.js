// DeepCube in the browser: the trained cost-to-go network (exported to ONNX from
// https://github.com/kjjoe/rubiks, deepcube/export_web.py) guides a batch weighted A* search,
// the same algorithm as deepcube/search.py. Needs onnxruntime-web loaded as the global `ort`.

import { FACE_MOVES, movePerm, isSolved } from "./cube.js";

const PERMS = FACE_MOVES.map(movePerm);
const keyOf = (s) => String.fromCharCode(...s);

// binary min-heap of [f, tie, key, g]
class Heap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  less(i, j) { const x = this.a[i], y = this.a[j]; return x[0] < y[0] || (x[0] === y[0] && x[1] < y[1]); }
  swap(i, j) { [this.a[i], this.a[j]] = [this.a[j], this.a[i]]; }
  push(item) {
    this.a.push(item);
    for (let i = this.a.length - 1; i > 0;) {
      const p = (i - 1) >> 1;
      if (!this.less(i, p)) break;
      this.swap(i, p);
      i = p;
    }
  }
  pop() {
    const top = this.a[0], last = this.a.pop();
    if (this.a.length) {
      this.a[0] = last;
      for (let i = 0; ;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < this.a.length && this.less(l, m)) m = l;
        if (r < this.a.length && this.less(r, m)) m = r;
        if (m === i) break;
        this.swap(i, m);
        i = m;
      }
    }
    return top;
  }
}

export class DeepCube {
  constructor(session, backend) {
    this.session = session;
    this.backend = backend;
  }

  // Try WebGPU first (fast); fall back to WebAssembly on the CPU.
  static async load(url) {
    try {
      if (!navigator.gpu) throw new Error("no WebGPU");
      const s = await ort.InferenceSession.create(url, { executionProviders: ["webgpu"] });
      return new DeepCube(s, "WebGPU");
    } catch (err) {
      console.warn("DeepCube: WebGPU unavailable, using WebAssembly", err);
      const s = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
      return new DeepCube(s, "WebAssembly");
    }
  }

  // estimated moves to solved for each 54-sticker state
  async predict(states) {
    const x = new Float32Array(states.length * 324);
    states.forEach((s, n) => {
      for (let i = 0; i < 54; i++) x[n * 324 + i * 6 + s[i]] = 1;
    });
    const out = await this.session.run({ x: new ort.Tensor("float32", x, [states.length, 324]) });
    return out.cost.data;
  }

  // f = weight * g + h; expands `batch` states per network call
  async solve(start, { weight = 0.6, batch = 100, timeLimit = 30000, maxNodes = 2e6, onProgress } = {}) {
    const t0 = performance.now();
    if (isSolved(start)) return { moves: [], solved: true, nodes: 1, seconds: 0 };

    const nodes = new Map(); // key -> { g, parent, move, state }
    const k0 = keyOf(start);
    nodes.set(k0, { g: 0, parent: null, move: -1, state: start });
    const heap = new Heap();
    let tie = 0;
    heap.push([(await this.predict([start]))[0], tie++, k0, 0]);

    const pathTo = (k) => {
      const moves = [];
      for (let n = nodes.get(k); n.parent !== null; n = nodes.get(n.parent)) moves.push(FACE_MOVES[n.move]);
      return moves.reverse();
    };
    const result = (moves, solved) => ({ moves, solved, nodes: nodes.size, seconds: (performance.now() - t0) / 1000 });

    while (heap.size) {
      if (performance.now() - t0 > timeLimit || nodes.size > maxNodes) break;

      // pop the best states, skipping entries made stale by a shorter path found later
      const popped = [];
      while (heap.size && popped.length < batch) {
        const [, , k, g] = heap.pop();
        if (nodes.get(k).g === g) popped.push(k);
      }

      const newKeys = [], newStates = [], newG = [];
      for (const pk of popped) {
        const parent = nodes.get(pk);
        const g = parent.g + 1;
        for (let m = 0; m < 18; m++) {
          const p = PERMS[m];
          const s = new Uint8Array(54);
          for (let i = 0; i < 54; i++) s[i] = parent.state[p[i]];
          if (isSolved(s)) return result([...pathTo(pk), FACE_MOVES[m]], true);
          const k = keyOf(s);
          const old = nodes.get(k);
          if (!old || g < old.g) {
            nodes.set(k, { g, parent: pk, move: m, state: s });
            newKeys.push(k);
            newStates.push(s);
            newG.push(g);
          }
        }
      }
      if (!newKeys.length) continue;

      const h = await this.predict(newStates);
      for (let i = 0; i < newKeys.length; i++) heap.push([weight * newG[i] + h[i], tie++, newKeys[i], newG[i]]);
      if (onProgress) onProgress({ nodes: nodes.size, seconds: (performance.now() - t0) / 1000, best: Math.min(...h) });
    }
    return result([], false);
  }
}
