const N = 120;
const iter = 4;
const SCALE = 6;
const dt = 0.1;

let s, density, Vx, Vy, Vx0, Vy0, obstacles;

function IX(x, y) {
    x = Math.max(0, Math.min(x, N - 1));
    y = Math.max(0, Math.min(y, N - 1));
    return x + y * N;
}

function setupFluid() {
    let size = N * N;
    s = new Float32Array(size);
    density = new Float32Array(size);
    Vx = new Float32Array(size);
    Vy = new Float32Array(size);
    Vx0 = new Float32Array(size);
    Vy0 = new Float32Array(size);
    obstacles = new Uint8Array(size);
}

function set_bnd(b, x) {
    for (let i = 1; i < N - 1; i++) {
        x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(N - 1, i)] = b == 1 ? -x[IX(N - 2, i)] : x[IX(N - 2, i)];
        x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N - 1)] = b == 2 ? -x[IX(i, N - 2)] : x[IX(i, N - 2)];
    }
    x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, N - 1)] = 0.5 * (x[IX(1, N - 1)] + x[IX(0, N - 2)]);
    x[IX(N - 1, 0)] = 0.5 * (x[IX(N - 2, 0)] + x[IX(N - 1, 1)]);
    x[IX(N - 1, N - 1)] = 0.5 * (x[IX(N - 2, N - 1)] + x[IX(N - 1, N - 2)]);

    for (let j = 1; j < N - 1; j++) {
        for (let i = 1; i < N - 1; i++) {
            if (obstacles[IX(i, j)]) {
                x[IX(i, j)] = 0;
            }
        }
    }
}

function lin_solve(b, x, x0, a, c) {
    let cRecip = 1.0 / c;
    for (let k = 0; k < iter; k++) {
        for (let j = 1; j < N - 1; j++) {
            for (let i = 1; i < N - 1; i++) {
                if (obstacles[IX(i, j)]) continue;
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i + 1, j)] + x[IX(i - 1, j)] + x[IX(i, j + 1)] + x[IX(i, j - 1)])) * cRecip;
            }
        }
        set_bnd(b, x);
    }
}

function advect(b, d, d0, u, v, dt) {
    let dt0 = dt * (N - 2);
    for (let j = 1; j < N - 1; j++) {
        for (let i = 1; i < N - 1; i++) {
            if (obstacles[IX(i, j)]) continue;

            let x = i - dt0 * u[IX(i, j)]; 
            let y = j - dt0 * v[IX(i, j)];

            if (x < 0.5) x = 0.5; 
            if (x > N - 1.5) x = N - 1.5;
            let i0 = Math.floor(x); let i1 = i0 + 1.0;
            if (y < 0.5) y = 0.5; 
            if (y > N - 1.5) y = N - 1.5;
            let j0 = Math.floor(y); let j1 = j0 + 1.0;

            let s1 = x - i0; let s0 = 1.0 - s1; 
            let t1 = y - j0; let t0 = 1.0 - t1;

            d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                          s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
    }
    set_bnd(b, d);
}

function project(u, v, p, div) {
    let h = 1.0 / N;
    for (let j = 1; j < N - 1; j++) {
        for (let i = 1; i < N - 1; i++) {
            if (obstacles[IX(i, j)]) {
                div[IX(i,j)] = 0;
                p[IX(i,j)] = 0;
                continue;
            }
            div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(0, div); set_bnd(0, p);
    lin_solve(0, p, div, 1, 4);

    for (let j = 1; j < N - 1; j++) {
        for (let i = 1; i < N - 1; i++) {
            if (obstacles[IX(i, j)]) continue;
            u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
            v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
        }
    }
    set_bnd(1, u); set_bnd(2, v);
}

function stepVelocity(u, v, u0, v0, dt) {
    for (let i = 0; i < N * N; i++) {
        u[i] += u0[i] * dt;
        v[i] += v0[i] * dt;
        u0[i] = 0; v0[i] = 0;
    }
    project(u, v, u0, v0);
    advect(1, u0, u, u, v, dt);
    advect(2, v0, v, u, v, dt);
    project(u0, v0, u, v);
    for (let i = 0; i < N * N; i++) {
        u[i] = u0[i];
        v[i] = v0[i];
    }
}

function stepDensity(x, x0, u, v, dt) {
    for (let i = 0; i < N * N; i++) {
        x[i] += x0[i] * dt;
        x0[i] = 0; 
    }
    advect(0, x0, x, u, v, dt);
    for (let i = 0; i < N * N; i++) {
        x[i] = x0[i];
    }
}

let canvas, ctx, isDrawing = false, isErasing = false;

function initApp() {
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext("2d");
    canvas.width = N * SCALE;
    canvas.height = N * SCALE;
    setupFluid();
    
    // Default obstacle
    for(let i=30; i<45; i++) {
        for(let j=50; j<70; j++) {
            obstacles[IX(i, j)] = 1;
        }
    }

    canvas.addEventListener('mousedown', e => { 
        isDrawing = true; 
        isErasing = e.button === 2; 
        drawObstacle(e);
    });
    canvas.addEventListener('mouseup', () => { isDrawing = false; isErasing = false; });
    canvas.addEventListener('mousemove', e => { if (isDrawing) drawObstacle(e); });
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    
    requestAnimationFrame(render);
}

function drawObstacle(e) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / SCALE);
    const y = Math.floor((e.clientY - rect.top) / SCALE);
    const radius = 3;
    for (let i = -radius; i <= radius; i++) {
        for (let j = -radius; j <= radius; j++) {
            if (i*i + j*j <= radius*radius) {
                let idx = IX(x+i, y+j);
                obstacles[idx] = isErasing ? 0 : 1;
                density[idx] = 0;
            }
        }
    }
}

function render() {
    let windSpeed = parseFloat(document.getElementById('windSpeed').value);
    
    // Continual injections across the inlet
    for (let j = 20; j < N - 20; j++) {
        Vx0[IX(2, j)] = windSpeed;
        if (j % 5 < 2) {
            s[IX(2, j)] = 255;
        }
    }
    
    stepVelocity(Vx, Vy, Vx0, Vy0, dt);
    stepDensity(density, s, Vx, Vy, dt);
    
    ctx.fillStyle = "#0c0d11";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            let idx = IX(i, j);
            if (obstacles[idx]) {
                ctx.fillStyle = "#efefef";
                ctx.fillRect(i * SCALE, j * SCALE, SCALE, SCALE);
            } else {
                let d = density[idx];
                if (d > 1) {
                    let alpha = Math.min(d / 200, 1.0);
                    ctx.fillStyle = `rgba(0, 200, 255, ${alpha})`;
                    ctx.fillRect(i * SCALE, j * SCALE, SCALE, SCALE);
                }
            }
        }
    }
    requestAnimationFrame(render);
}

window.onload = initApp;
