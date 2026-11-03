// Interactive Rubik's cube for the DeepCube post: a Three.js cube you can turn, scramble and
// solve with either DeepCube or the beginner-method solver.
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { STICKERS, SOLVED, movePerm, applyMove, isSolved, solveBeginner, randomScramble } from "./cube.js";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.js";
const MODEL_URL = new URL("./deepcube.onnx", import.meta.url).href;
const COLORS = [0xf4f4f4, 0xffd500, 0xc41e3a, 0xff6a00, 0x0051ba, 0x009e60]; // white yellow red orange blue green
const AXES = ["x", "y", "z"];

// ---------------------------------------------------------------------------
// Geometry: where each of the 54 stickers sits in 3D
// ---------------------------------------------------------------------------

// The MATLAB array has U at k = 0, F at i = 4 and R at j = 4. Map that to Three.js with
// U facing +y, F facing the camera (+z) and R on the right (+x).
const STICKER_POS = STICKERS.map((idx) => {
  const i = idx % 5, j = Math.floor(idx / 5) % 5, k = Math.floor(idx / 25);
  return [j - 2, 2 - k, i - 2]; // one coordinate is +-2: the face the sticker is on
});
const clamp1 = (v) => Math.max(-1, Math.min(1, v));

// integer rotation of a position by a multiple of 90 degrees (right-handed, like Three.js)
function rotate([x, y, z], axis, quarters) {
  for (let q = 0; q < ((quarters % 4) + 4) % 4; q++) {
    if (axis === 0) [y, z] = [-z, y];
    else if (axis === 1) [x, z] = [z, -x];
    else [x, y] = [-y, x];
  }
  return [x, y, z];
}

// For each move: the rotation axis, angle and layer, found from its permutation so the
// animation always agrees with the cube logic.
const motionCache = new Map();
function motion(name) {
  if (motionCache.has(name)) return motionCache.get(name);
  const p = movePerm(name);
  const moved = [...p.keys()].filter((s) => p[s] !== s);
  for (let axis = 0; axis < 3; axis++) {
    for (const quarters of [1, -1, 2]) {
      const ok = moved.every((s) => rotate(STICKER_POS[p[s]], axis, quarters).join() === STICKER_POS[s].join());
      if (ok) {
        const layer = new Set(moved.map((s) => clamp1(STICKER_POS[s][axis])));
        const m = { axis, angle: (quarters * Math.PI) / 2, layer };
        motionCache.set(name, m);
        return m;
      }
    }
  }
  throw new Error(`no rotation matches move ${name}`);
}

const inverse = (m) => (m[1] === "2" ? m : m[1] === "'" ? m[0] : m + "'");

// ---------------------------------------------------------------------------
// Widget
// ---------------------------------------------------------------------------

class CubeWidget {
  constructor(root) {
    this.root = root;
    this.state = SOLVED.slice();
    this.queue = [];
    this.anim = null;
    this.solution = null; // { moves, pos }
    this.playing = false;
    this.busy = false;
    this.deepcube = null;
    this.buildScene();
    this.buildControls();
    this.recolor();
    this.render();
  }

  // --- scene ---

  buildScene() {
    const stage = this.root.querySelector(".rubiks-stage");
    this.canvas = stage.querySelector("canvas");
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
    this.camera.position.set(0, 0, 9);
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(3, 5, 6);
    this.scene.add(sun);

    this.world = new THREE.Group();
    this.world.rotation.set(0.5, -0.65, 0);
    this.scene.add(this.world);

    const box = new THREE.BoxGeometry(0.96, 0.96, 0.96);
    const body = new THREE.MeshStandardMaterial({ color: 0x0c0e12, roughness: 0.6 });
    const shape = new THREE.Shape();
    const h = 0.4, r = 0.08;
    shape.moveTo(-h + r, -h);
    shape.lineTo(h - r, -h); shape.quadraticCurveTo(h, -h, h, -h + r);
    shape.lineTo(h, h - r); shape.quadraticCurveTo(h, h, h - r, h);
    shape.lineTo(-h + r, h); shape.quadraticCurveTo(-h, h, -h, h - r);
    shape.lineTo(-h, -h + r); shape.quadraticCurveTo(-h, -h, -h + r, -h);
    const tile = new THREE.ShapeGeometry(shape);

    this.cubies = new Map(); // "x,y,z" -> group
    for (let x = -1; x <= 1; x++) for (let y = -1; y <= 1; y++) for (let z = -1; z <= 1; z++) {
      if (!x && !y && !z) continue;
      const g = new THREE.Group();
      g.position.set(x, y, z);
      g.userData.home = [x, y, z];
      g.add(new THREE.Mesh(box, body));
      this.world.add(g);
      this.cubies.set([x, y, z].join(), g);
    }
    this.stickerMeshes = STICKER_POS.map((pos) => {
      const c = pos.map(clamp1);
      const normal = new THREE.Vector3(...pos.map((v, a) => (Math.abs(v) === 2 ? Math.sign(v) : 0)));
      const mesh = new THREE.Mesh(tile, new THREE.MeshStandardMaterial({ roughness: 0.35 }));
      mesh.position.copy(normal).multiplyScalar(0.485);
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
      this.cubies.get(c.join()).add(mesh);
      return mesh;
    });

    const resize = () => {
      const w = stage.clientWidth, hgt = stage.clientHeight;
      this.renderer.setSize(w, hgt, false);
      this.camera.aspect = w / hgt;
      this.camera.updateProjectionMatrix();
      this.render();
    };
    new ResizeObserver(resize).observe(stage);
    resize();

    // drag to look around
    let drag = null;
    this.canvas.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY };
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.world.rotation.y += (e.clientX - drag.x) * 0.01;
      this.world.rotation.x = Math.max(-1.4, Math.min(1.4, this.world.rotation.x + (e.clientY - drag.y) * 0.01));
      drag = { x: e.clientX, y: e.clientY };
      this.render();
    });
    const end = () => { drag = null; };
    this.canvas.addEventListener("pointerup", end);
    this.canvas.addEventListener("pointercancel", end);
  }

  render() {
    if (this.frame) return;
    this.frame = requestAnimationFrame((t) => {
      this.frame = null;
      this.step(t);
      this.renderer.render(this.scene, this.camera);
    });
  }

  recolor() {
    this.stickerMeshes.forEach((m, s) => m.material.color.setHex(COLORS[this.state[s]]));
  }

  // --- turning ---

  enqueue(moves, ms) {
    for (const m of moves) this.queue.push({ m, ms });
    this.render();
  }

  get idle() {
    return !this.anim && !this.queue.length;
  }

  step(t) {
    if (!this.anim && this.queue.length) {
      const { m, ms, onDone } = this.queue.shift();
      const { axis, angle, layer } = motion(m);
      const pivot = new THREE.Group();
      this.world.add(pivot);
      for (const g of this.cubies.values()) if (layer.has(g.userData.home[axis])) pivot.attach(g);
      this.anim = { m, pivot, axis, angle, start: t, ms: m[1] === "2" ? ms * 1.5 : ms, onDone };
    }
    if (!this.anim) {
      this.onIdle?.();
      return;
    }
    const a = this.anim;
    const u = Math.min(1, (t - a.start) / a.ms);
    a.pivot.rotation[AXES[a.axis]] = a.angle * (u < 0.5 ? 2 * u * u : 1 - 2 * (1 - u) * (1 - u));
    if (u >= 1) {
      // snap the cubies home and show the turn by recolouring the stickers
      for (const g of [...a.pivot.children]) {
        this.world.attach(g);
        g.position.set(...g.userData.home);
        g.rotation.set(0, 0, 0);
      }
      this.world.remove(a.pivot);
      this.state = applyMove(this.state, a.m);
      this.recolor();
      this.anim = null;
      this.afterTurn?.(a.m);
    }
    this.render();
  }

  // --- controls ---

  $(sel) { return this.root.querySelector(sel); }

  buildControls() {
    this.root.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b || b.disabled) return;
      if (b.dataset.move) this.manual(b.dataset.move);
      else this.action(b.dataset.act);
    });
    this.root.addEventListener("keydown", (e) => {
      const f = e.key.toUpperCase();
      if ("UDLRFB".includes(f) && f.length === 1 && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        this.manual(e.shiftKey ? f + "'" : f);
      } else if (e.key === " ") {
        e.preventDefault();
        this.action("play");
      }
    });
    this.speed = this.$(".rubiks-speed");
    this.status = this.$(".rubiks-status");
    this.movesEl = this.$(".rubiks-moves");
    this.afterTurn = () => this.updatePlayback();
    this.onIdle = () => {
      if (this.playing && this.solution && this.solution.pos < this.solution.moves.length) this.playNext();
      else if (this.playing) { this.playing = false; this.updatePlayback(); }
    };
    this.updatePlayback();
  }

  ms() {
    return 600 - Number(this.speed.value) * 5.5; // slider 0-100 -> 600-50 ms per quarter turn
  }

  manual(m) {
    if (this.busy || !this.idle) return;
    this.clearSolution();
    this.enqueue([m], this.ms());
  }

  setBusy(busy) {
    this.busy = busy;
    this.root.querySelectorAll("button, select, input").forEach((el) => {
      if (!el.closest(".rubiks-playback")) el.disabled = busy;
    });
    this.updatePlayback();
  }

  say(html) {
    this.status.innerHTML = html;
  }

  async action(act) {
    if (act === "scramble") {
      if (this.busy || !this.idle) return;
      this.clearSolution();
      const n = Number(this.$(".rubiks-length").value);
      const moves = randomScramble(n);
      this.say(`Scramble (${n} moves): <code>${moves.join(" ")}</code>`);
      this.enqueue(moves, 70);
    } else if (act === "reset") {
      if (this.busy || !this.idle) return;
      this.clearSolution();
      this.state = SOLVED.slice();
      this.recolor();
      this.say("");
      this.render();
    } else if (act === "solve") {
      if (this.busy || !this.idle) return;
      await this.solve();
    } else if (act === "play") {
      if (!this.solution) return;
      if (this.playing) { this.playing = false; }
      else if (this.solution.pos < this.solution.moves.length) { this.playing = true; this.render(); }
      this.updatePlayback();
    } else if (act === "next") {
      if (this.solution && !this.playing && this.idle && this.solution.pos < this.solution.moves.length) this.playNext();
    } else if (act === "prev") {
      if (this.solution && !this.playing && this.idle && this.solution.pos > 0) {
        this.solution.pos--;
        this.enqueue([inverse(this.solution.moves[this.solution.pos])], this.ms());
      }
    } else if (act === "restart") {
      if (!this.solution || this.playing || !this.idle) return;
      const back = this.solution.moves.slice(0, this.solution.pos).reverse().map(inverse);
      this.solution.pos = 0;
      for (const m of back) this.state = applyMove(this.state, m);
      this.recolor();
      this.updatePlayback();
      this.render();
    }
  }

  playNext() {
    const s = this.solution;
    this.enqueue([s.moves[s.pos]], this.ms());
    s.pos++;
    this.updatePlayback();
  }

  async solve() {
    if (isSolved(this.state)) {
      this.say("The cube is already solved. Scramble it first.");
      return;
    }
    const which = this.$("input[name=rubiks-solver]:checked").value;
    this.setBusy(true);
    try {
      if (which === "beginner") {
        const t0 = performance.now();
        const moves = solveBeginner(this.state);
        this.say(`Beginner method: <b>${moves.length} moves</b> (found in ${((performance.now() - t0) / 1000).toFixed(2)} s).`);
        this.showSolution(moves);
      } else {
        if (!this.deepcube) {
          this.say("Loading the DeepCube network (7 MB)…");
          this.deepcube = await loadDeepCube();
        }
        const r = await this.deepcube.solve(this.state, {
          onProgress: ({ nodes, seconds }) => this.say(`DeepCube is searching… ${nodes.toLocaleString()} positions checked (${seconds.toFixed(1)} s)`),
        });
        const how = `${r.nodes.toLocaleString()} positions checked in ${r.seconds.toFixed(1)} s on ${this.deepcube.backend}`;
        if (r.solved) {
          this.say(`DeepCube: <b>${r.moves.length} moves</b> (${how}).`);
          this.showSolution(r.moves);
        } else {
          this.say(`DeepCube couldn't find a solution in time (${how}). The current network only sees about 10 moves ahead, so try a shorter scramble or the beginner method.`);
        }
      }
    } catch (err) {
      console.error(err);
      this.say(`Something went wrong: ${err.message}`);
    }
    this.setBusy(false);
  }

  showSolution(moves) {
    this.solution = { moves, pos: 0 };
    this.movesEl.innerHTML = moves.map((m, i) => `<span data-i="${i}">${m}</span>`).join(" ");
    this.playing = true;
    this.updatePlayback();
    this.render();
  }

  clearSolution() {
    this.solution = null;
    this.playing = false;
    this.movesEl.innerHTML = "";
    this.updatePlayback();
  }

  updatePlayback() {
    const s = this.solution;
    const bar = this.$(".rubiks-playback");
    bar.hidden = !s;
    if (!s) return;
    this.$("[data-act=play]").textContent = this.playing ? "Pause" : "Play";
    this.$("[data-act=play]").disabled = !this.playing && s.pos >= s.moves.length;
    this.$("[data-act=next]").disabled = this.playing || s.pos >= s.moves.length;
    this.$("[data-act=prev]").disabled = this.playing || s.pos === 0;
    this.$("[data-act=restart]").disabled = this.playing || s.pos === 0;
    this.movesEl.querySelectorAll("span").forEach((el) => {
      el.className = Number(el.dataset.i) < s.pos ? "done" : Number(el.dataset.i) === s.pos ? "next" : "";
    });
  }
}

// onnxruntime-web is only downloaded the first time DeepCube is used
async function loadDeepCube() {
  if (!window.ort) {
    await new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = ORT_URL;
      s.onload = resolve;
      s.onerror = () => reject(new Error("could not load onnxruntime-web"));
      document.head.appendChild(s);
    });
  }
  const { DeepCube } = await import("./deepcube.js");
  return DeepCube.load(MODEL_URL);
}

document.querySelectorAll(".rubiks").forEach((el) => new CubeWidget(el));
