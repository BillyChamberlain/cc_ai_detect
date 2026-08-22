const canvas = document.querySelector("#canvas");
const ctx = canvas.getContext("2d");

const phaseTitle = document.querySelector("#phaseTitle");
const phaseDescription = document.querySelector("#phaseDescription");
const instruction = document.querySelector("#instruction");

const TAU = Math.PI * 2;

const timeline = {
  seedEnd: 15,
  orbitStart: 15,
  interactionStart: 25,
  atmosphereStart: 40
};

let width = 0;
let height = 0;
let dpr = 1;
let center = { x: 0, y: 0 };
let startedAt = performance.now();

let stars = [];
let veins = [];
let bodies = [];

const palette = {
  background: "#05060d",
  shell: "rgba(177, 222, 255, 0.84)",
  shellSoft: "rgba(125, 195, 255, 0.28)",
  vein: "rgba(180, 229, 255, 0.92)"
};

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function easeInOut(value) {
  return value < 0.5
    ? 2 * value * value
    : 1 - Math.pow(-2 * value + 2, 2) / 2;
}

function randomRange(min, max) {
  return min + Math.random() * (max - min);
}

function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  width = window.innerWidth;
  height = window.innerHeight;

  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  center = {
    x: width / 2,
    y: height / 2
  };

  createStars();
}

function createStars() {
  const count = Math.floor((width * height) / 7000);

  stars = Array.from({ length: count }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    radius: Math.random() * 0.85 + 0.12,
    alpha: Math.random() * 0.45 + 0.08,
    pulse: Math.random() * TAU,
    drift: 0.15 + Math.random() * 0.4
  }));
}

function setPhaseUI(time) {
  if (time < timeline.seedEnd) {
    phaseTitle.textContent = "SIGIL SEED";
    phaseDescription.textContent =
      "A repeatable shell discovers its inside-out form.";
    instruction.textContent = "The form is drawing itself.";
  } else if (time < timeline.interactionStart) {
    phaseTitle.textContent = "BEHAVIOR RULE";
    phaseDescription.textContent =
      "Nested shells orbit an inner light. No surfaces. Only thresholds.";
    instruction.textContent = "Observe the shell-orbits.";
  } else if (time < timeline.atmosphereStart) {
    phaseTitle.textContent = "GESTURE LANGUAGE";
    phaseDescription.textContent =
      "Click to split a signal into branching right-angle veins. Their retreat forms new orbiting shells.";
    instruction.textContent = "Click anywhere to fracture the field.";
  } else {
    phaseTitle.textContent = "ATMOSPHERE CONSTRAINT";
    phaseDescription.textContent =
      "Heavy grey fog conceals and protects the system beneath it.";
    instruction.textContent = "Click to grow another orbital cluster.";
  }
}

function drawBackground(time) {
  const gradient = ctx.createRadialGradient(
    center.x,
    center.y,
    0,
    center.x,
    center.y,
    Math.max(width, height) * 0.8
  );

  gradient.addColorStop(0, "#0b1324");
  gradient.addColorStop(0.38, "#080d19");
  gradient.addColorStop(1, "#03040a");

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  const starOpacity = 0.28 + clamp(time / 42) * 0.52;

  for (const star of stars) {
    const flicker =
      0.45 + 0.55 * Math.sin(time * star.drift + star.pulse);

    ctx.beginPath();
    ctx.fillStyle = `rgba(186, 218, 255, ${
      star.alpha * flicker * starOpacity
    })`;
    ctx.arc(star.x, star.y, star.radius, 0, TAU);
    ctx.fill();
  }
}

function drawSigilSeed(time) {
  const progress = clamp(time / timeline.seedEnd);
  const drawProgress = easeInOut(clamp(progress * 1.35));
  const inversion = easeInOut(clamp((progress - 0.3) / 0.7));

  const baseRadius = Math.min(width, height) * 0.17;
  const pulse = Math.sin(time * 1.8) * 3;
  const ringCount = 13;

  ctx.save();
  ctx.translate(center.x, center.y);
  ctx.rotate(time * 0.07);

  for (let ring = 0; ring < ringCount; ring++) {
    const ringProgress = ring / (ringCount - 1);
    const visible = clamp((drawProgress - ringProgress * 0.18) / 0.45);

    if (visible <= 0) continue;

    const radius =
      baseRadius *
      (0.2 + ringProgress * 1.15) *
      (1 - inversion * 0.52 + ringProgress * inversion * 0.85);

    const wobble =
      6 +
      ringProgress * 16 +
      Math.sin(time * 0.8 + ring * 0.8) * 3;

    ctx.beginPath();

    const points = 180;
    const end = Math.floor(points * visible);

    for (let i = 0; i <= end; i++) {
      const theta = (i / points) * TAU;
      const fold =
        Math.sin(theta * 3 + time * 0.8 + ring) *
        wobble *
        (0.25 + inversion);

      const ripple =
        Math.sin(theta * 7 - time * 0.45 + ring * 0.7) *
        (2 + ringProgress * 7);

      const localRadius =
        radius +
        fold +
        ripple +
        pulse * (1 - ringProgress * 0.5);

      const x = Math.cos(theta) * localRadius;
      const y = Math.sin(theta) * localRadius;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.strokeStyle = `rgba(157, 214, 255, ${
      0.06 + visible * (0.18 + ringProgress * 0.16)
    })`;

    ctx.lineWidth = 0.45 + ringProgress * 0.8;
    ctx.stroke();
  }

  ctx.globalCompositeOperation = "lighter";

  for (let spoke = 0; spoke < 72; spoke++) {
    const angle = (spoke / 72) * TAU + time * 0.13;
    const reach =
      baseRadius *
      (0.75 + 0.35 * Math.sin(spoke * 1.7 + time * 1.1)) *
      drawProgress;

    const inner =
      baseRadius *
      0.14 *
      (1 - inversion * 0.7) *
      (0.7 + 0.3 * Math.sin(spoke));

    ctx.beginPath();
    ctx.moveTo(Math.cos(angle) * inner, Math.sin(angle) * inner);
    ctx.lineTo(Math.cos(angle) * reach, Math.sin(angle) * reach);
    ctx.strokeStyle = `rgba(172, 230, 255, ${
      0.02 + drawProgress * 0.06
    })`;
    ctx.lineWidth = 0.35;
    ctx.stroke();
  }

  const coreRadius = baseRadius * (0.22 + inversion * 0.1);
  const coreGlow = ctx.createRadialGradient(0, 0, 0, 0, 0, coreRadius * 3);

  coreGlow.addColorStop(0, "rgba(225, 245, 255, 0.86)");
  coreGlow.addColorStop(0.16, "rgba(129, 207, 255, 0.42)");
  coreGlow.addColorStop(1, "rgba(90, 169, 255, 0)");

  ctx.fillStyle = coreGlow;
  ctx.beginPath();
  ctx.arc(0, 0, coreRadius * 3, 0, TAU);
  ctx.fill();

  ctx.restore();
  ctx.globalCompositeOperation = "source-over";
}

function createDefaultBodies() {
  if (bodies.length > 0) return;

  bodies = [
    {
      radius: 18,
      orbitRadius: Math.min(width, height) * 0.18,
      speed: 0.34,
      angle: 0.6,
      tilt: -0.28,
      shellCount: 4,
      birth: 0,
      emergeDuration: 0,
      moons: [{ radius: 5, orbitRadius: 34, speed: 1.5, angle: 0.4 }]
    },
    {
      radius: 30,
      orbitRadius: Math.min(width, height) * 0.29,
      speed: -0.2,
      angle: 2.8,
      tilt: 0.5,
      shellCount: 5,
      birth: 0,
      emergeDuration: 0,
      moons: [
        { radius: 5.5, orbitRadius: 46, speed: -1.1, angle: 2.2 },
        { radius: 3.5, orbitRadius: 32, speed: 1.9, angle: 4.8 }
      ]
    },
    {
      radius: 13,
      orbitRadius: Math.min(width, height) * 0.38,
      speed: 0.12,
      angle: 5,
      tilt: -0.12,
      shellCount: 4,
      birth: 0,
      emergeDuration: 0,
      moons: []
    }
  ];
}

function orbitPosition(body, time) {
  const angle = body.angle + time * body.speed;
  const orbitScaleY = 0.45;

  return {
    x: center.x + Math.cos(angle) * body.orbitRadius,
    y: center.y + Math.sin(angle) * body.orbitRadius * orbitScaleY
  };
}

function drawSun(time, opacity) {
  const radius = Math.min(width, height) * 0.047;
  const pulse = 1 + Math.sin(time * 2.2) * 0.05;

  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  const glow = ctx.createRadialGradient(
    center.x,
    center.y,
    0,
    center.x,
    center.y,
    radius * 7
  );

  glow.addColorStop(0, `rgba(232, 246, 255, ${0.95 * opacity})`);
  glow.addColorStop(0.12, `rgba(168, 220, 255, ${0.65 * opacity})`);
  glow.addColorStop(0.42, `rgba(92, 170, 255, ${0.14 * opacity})`);
  glow.addColorStop(1, "rgba(92, 170, 255, 0)");

  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius * 7 * pulse, 0, TAU);
  ctx.fill();

  ctx.strokeStyle = `rgba(222, 243, 255, ${0.8 * opacity})`;
  ctx.lineWidth = 0.7;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius * pulse, 0, TAU);
  ctx.stroke();

  ctx.restore();
}

function drawOrbit(body, time, opacity) {
  ctx.save();
  ctx.translate(center.x, center.y);
  ctx.rotate(body.tilt);

  ctx.strokeStyle = `rgba(135, 185, 232, ${0.11 * opacity})`;
  ctx.lineWidth = 0.5;
  ctx.setLineDash([2, 8]);
  ctx.lineDashOffset = -time * 7;

  ctx.beginPath();
  ctx.ellipse(
    0,
    0,
    body.orbitRadius,
    body.orbitRadius * 0.45,
    0,
    0,
    TAU
  );
  ctx.stroke();

  ctx.restore();
  ctx.setLineDash([]);
}

function drawShell(x, y, radius, time, shellCount, opacity = 1, seed = 1) {
  ctx.save();
  ctx.translate(x, y);
  ctx.globalCompositeOperation = "lighter";

  const glow = ctx.createRadialGradient(0, 0, 0, 0, 0, radius * 2.8);
  glow.addColorStop(0, `rgba(139, 207, 255, ${0.11 * opacity})`);
  glow.addColorStop(0.5, `rgba(93, 170, 255, ${0.04 * opacity})`);
  glow.addColorStop(1, "rgba(93, 170, 255, 0)");

  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(0, 0, radius * 2.8, 0, TAU);
  ctx.fill();

  for (let ring = 0; ring < shellCount; ring++) {
    const amount = ring / Math.max(shellCount - 1, 1);
    const ringRadius = radius * (0.38 + amount * 0.72);
    const points = 80;

    ctx.beginPath();

    for (let i = 0; i <= points; i++) {
      const angle = (i / points) * TAU;

      const irregularity =
        Math.sin(angle * 3 + time * (0.8 + amount) + seed * 1.7) *
          radius *
          (0.035 + amount * 0.045) +
        Math.sin(angle * 7 - time * 0.6 + seed) * radius * 0.018;

      const xPos = Math.cos(angle) * (ringRadius + irregularity);
      const yPos = Math.sin(angle) * (ringRadius * 0.72 + irregularity);

      if (i === 0) ctx.moveTo(xPos, yPos);
      else ctx.lineTo(xPos, yPos);
    }

    ctx.strokeStyle = `rgba(184, 229, 255, ${
      (0.16 + amount * 0.36) * opacity
    })`;

    ctx.lineWidth = 0.45 + amount * 0.55;
    ctx.stroke();
  }

  for (let line = 0; line < 7; line++) {
    const rotation = (line / 7) * Math.PI + time * 0.09;

    ctx.save();
    ctx.rotate(rotation);
    ctx.beginPath();
    ctx.ellipse(0, 0, radius * 0.87, radius * 0.26, 0, 0, TAU);
    ctx.strokeStyle = `rgba(152, 211, 255, ${0.09 * opacity})`;
    ctx.lineWidth = 0.38;
    ctx.stroke();
    ctx.restore();
  }

  ctx.restore();
  ctx.globalCompositeOperation = "source-over";
}

function drawOrbitalSystem(time, morph) {
  createDefaultBodies();

  const orbitOpacity = easeInOut(morph);

  drawSun(time, orbitOpacity);

  for (const body of bodies) {
    const emerge = body.emergeDuration
      ? easeInOut(clamp((time - body.birth) / body.emergeDuration))
      : 1;

    drawOrbit(body, time, orbitOpacity * emerge);
  }

  for (let index = 0; index < bodies.length; index++) {
    const body = bodies[index];
    const emerge = body.emergeDuration
      ? easeInOut(clamp((time - body.birth) / body.emergeDuration))
      : 1;

    const position = orbitPosition(body, time);
    const displayRadius = body.radius * emerge;

    if (displayRadius < 0.2) continue;

    drawShell(
      position.x,
      position.y,
      displayRadius,
      time,
      body.shellCount,
      orbitOpacity * emerge,
      index + 1
    );

    for (let moonIndex = 0; moonIndex < body.moons.length; moonIndex++) {
      const moon = body.moons[moonIndex];
      const moonAngle = moon.angle + time * moon.speed;

      const moonX = position.x + Math.cos(moonAngle) * moon.orbitRadius;
      const moonY = position.y + Math.sin(moonAngle) * moon.orbitRadius * 0.52;

      ctx.strokeStyle = `rgba(150, 207, 255, ${
        0.14 * orbitOpacity * emerge
      })`;

      ctx.lineWidth = 0.4;
      ctx.beginPath();
      ctx.ellipse(
        position.x,
        position.y,
        moon.orbitRadius,
        moon.orbitRadius * 0.52,
        0,
        0,
        TAU
      );
      ctx.stroke();

      drawShell(
        moonX,
        moonY,
        moon.radius * emerge,
        time,
        3,
        orbitOpacity * emerge,
        index * 10 + moonIndex + 5
      );
    }
  }
}

function buildBranch(startX, startY, startDirection, branchSeed) {
  const points = [{ x: startX, y: startY }];
  const margin = 18;
  const maxSteps = Math.floor(randomRange(85, 155));
  const stepSize = randomRange(4.8, 7.2);

  let direction = startDirection;
  let x = startX;
  let y = startY;

  const directions = [
    { x: 1, y: 0 },
    { x: 0, y: 1 },
    { x: -1, y: 0 },
    { x: 0, y: -1 }
  ];

  for (let i = 0; i < maxSteps; i++) {
    const turnChance = i < 10 ? 0.42 : 0.3;

    if (Math.random() < turnChance) {
      direction = (direction + (Math.random() < 0.5 ? 1 : 3)) % 4;
    }

    let nextX = x + directions[direction].x * stepSize;
    let nextY = y + directions[direction].y * stepSize;

    if (
      nextX < margin ||
      nextX > width - margin ||
      nextY < margin ||
      nextY > height - margin
    ) {
      direction = (direction + 2) % 4;
      nextX = x + directions[direction].x * stepSize;
      nextY = y + directions[direction].y * stepSize;
    }

    x = nextX;
    y = nextY;

    points.push({ x, y });

    if (i > 25 && Math.random() < 0.025) {
      break;
    }
  }

  return {
    points,
    seed: branchSeed,
    birth: performance.now() / 1000,
    growDuration: randomRange(1.35, 2.7),
    holdDuration: randomRange(0.9, 1.8),
    retreatDuration: randomRange(3.4, 5.1),
    morphDuration: randomRange(1.5, 2.7),
    sphereCreated: false,
    sphere: null
  };
}

function createVeinCluster(startX, startY) {
  const directions = [0, 1, 2, 3];
  const branchCount = Math.floor(randomRange(4, 7));

  for (let i = 0; i < branchCount; i++) {
    const primaryDirection = directions[i % directions.length];
    const branch = buildBranch(
      startX,
      startY,
      primaryDirection,
      Math.random() * 1000
    );

    veins.push(branch);
  }
}

function createMorphingSphere(vein, time) {
  const finalPoint = vein.points[vein.points.length - 1];
  const dx = finalPoint.x - center.x;
  const dy = finalPoint.y - center.y;

  const finalOrbitRadius = Math.max(
    Math.min(width, height) * 0.16,
    Math.min(Math.hypot(dx, dy / 0.45), Math.min(width, height) * 0.43)
  );

  const finalAngle = Math.atan2(dy / 0.45, dx);
  const radius = randomRange(10, 18);

  vein.sphere = {
    startX: finalPoint.x,
    startY: finalPoint.y,
    radius,
    orbitRadius: finalOrbitRadius,
    speed: randomRange(-0.32, 0.32) || 0.18,
    angle: finalAngle,
    tilt: randomRange(-0.65, 0.65),
    shellCount: 3,
    morphStart: time,
    morphDuration: vein.morphDuration,
    merged: false,
    seed: vein.seed
  };

  vein.sphereCreated = true;
}

function drawMorphingSphere(sphere, time) {
  const progress = easeInOut(
    clamp((time - sphere.morphStart) / sphere.morphDuration)
  );

  const targetPosition = orbitPosition(sphere, time);
  const x = sphere.startX + (targetPosition.x - sphere.startX) * progress;
  const y = sphere.startY + (targetPosition.y - sphere.startY) * progress;

  const formationRadius =
    sphere.radius * (0.18 + progress * 0.82);

  const shellOpacity = 0.3 + progress * 0.7;

  drawShell(
    x,
    y,
    formationRadius,
    time,
    sphere.shellCount,
    shellOpacity,
    sphere.seed
  );

  const pullLineAlpha = (1 - progress) * 0.32;

  if (pullLineAlpha > 0.01) {
    ctx.save();
    ctx.setLineDash([2, 7]);
    ctx.lineDashOffset = -time * 12;
    ctx.strokeStyle = `rgba(171, 222, 255, ${pullLineAlpha})`;
    ctx.lineWidth = 0.45;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(targetPosition.x, targetPosition.y);
    ctx.stroke();
    ctx.restore();
    ctx.setLineDash([]);
  }

  if (progress >= 1 && !sphere.merged) {
    bodies.push({
      radius: sphere.radius,
      orbitRadius: sphere.orbitRadius,
      speed: sphere.speed,
      angle: sphere.angle,
      tilt: sphere.tilt,
      shellCount: sphere.shellCount,
      birth: time,
      emergeDuration: 1.1,
      moons: []
    });

    sphere.merged = true;
  }
}

function drawVeins(time) {
  for (let index = veins.length - 1; index >= 0; index--) {
    const vein = veins[index];
    const age = time - vein.birth;

    const growEnd = vein.growDuration;
    const holdEnd = growEnd + vein.holdDuration;
    const retreatEnd = holdEnd + vein.retreatDuration;
    const morphEnd = retreatEnd + vein.morphDuration;

    if (age > morphEnd) {
      veins.splice(index, 1);
      continue;
    }

    const growAmount = clamp(age / growEnd);
    const retreatAmount =
      age <= holdEnd ? 0 : clamp((age - holdEnd) / vein.retreatDuration);

    const visibleStart = Math.floor(
      retreatAmount * vein.points.length * 0.88
    );

    const visibleEnd = Math.max(
      visibleStart + 1,
      Math.floor(growAmount * vein.points.length)
    );

    if (visibleEnd > visibleStart) {
      ctx.save();
      ctx.globalCompositeOperation = "lighter";

      ctx.beginPath();

      for (let p = visibleStart; p < visibleEnd; p++) {
        const point = vein.points[p];

        if (p === visibleStart) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      }

      const fading = 1 - retreatAmount * 0.7;

      ctx.strokeStyle = `rgba(170, 229, 255, ${0.82 * fading})`;
      ctx.lineWidth = 0.35 + (1 - retreatAmount) * 0.45;
      ctx.shadowColor = "rgba(104, 205, 255, 0.9)";
      ctx.shadowBlur = 9;
      ctx.stroke();

      ctx.shadowBlur = 0;

      const head = vein.points[
        Math.min(visibleEnd - 1, vein.points.length - 1)
      ];

      ctx.beginPath();
      ctx.fillStyle = `rgba(232, 249, 255, ${0.86 * fading})`;
      ctx.arc(head.x, head.y, 1.35, 0, TAU);
      ctx.fill();

      ctx.restore();
      ctx.globalCompositeOperation = "source-over";
    }

    if (!vein.sphereCreated && retreatAmount > 0.72) {
      createMorphingSphere(vein, time);
    }

    if (vein.sphereCreated && vein.sphere && !vein.sphere.merged) {
      drawMorphingSphere(vein.sphere, time);
    }
  }
}

function drawAtmosphere(time) {
  const intensity = easeInOut(
    clamp((time - timeline.atmosphereStart) / 10)
  );

  if (intensity <= 0) return;

  ctx.save();

  // Grey haze over the entire image.
  ctx.fillStyle = `rgba(84, 88, 94, ${0.2 * intensity})`;
  ctx.fillRect(0, 0, width, height);

  // Large moving banks of grey fog.
  const fogLayers = 11;

  for (let layer = 0; layer < fogLayers; layer++) {
    const horizontalDrift =
      Math.sin(time * (0.08 + layer * 0.004) + layer * 2.13) *
      width *
      0.17;

    const verticalDrift =
      Math.cos(time * 0.06 + layer * 1.8) *
      height *
      0.08;

    const x =
      width * (0.05 + (layer / fogLayers) * 0.95) + horizontalDrift;

    const y =
      height * (0.22 + ((layer * 0.19) % 0.7)) + verticalDrift;

    const radius = Math.max(width, height) * (0.28 + layer * 0.032);

    const fog = ctx.createRadialGradient(x, y, 0, x, y, radius);

    fog.addColorStop(0, `rgba(186, 187, 188, ${0.12 * intensity})`);
    fog.addColorStop(0.3, `rgba(138, 141, 145, ${0.09 * intensity})`);
    fog.addColorStop(0.67, `rgba(85, 88, 93, ${0.055 * intensity})`);
    fog.addColorStop(1, "rgba(45, 48, 54, 0)");

    ctx.fillStyle = fog;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, TAU);
    ctx.fill();
  }

  // A low, denser horizon fog.
  const horizon = ctx.createLinearGradient(0, height * 0.45, 0, height);

  horizon.addColorStop(0, "rgba(130, 133, 137, 0)");
  horizon.addColorStop(0.45, `rgba(135, 138, 142, ${0.12 * intensity})`);
  horizon.addColorStop(1, `rgba(46, 49, 54, ${0.34 * intensity})`);

  ctx.fillStyle = horizon;
  ctx.fillRect(0, height * 0.35, width, height * 0.65);

  // Stronger edge darkness preserves the ominous feeling.
  const vignette = ctx.createRadialGradient(
    center.x,
    center.y,
    Math.min(width, height) * 0.12,
    center.x,
    center.y,
    Math.max(width, height) * 0.78
  );

  vignette.addColorStop(0, "rgba(0, 0, 0, 0)");
  vignette.addColorStop(0.48, `rgba(0, 0, 0, ${0.13 * intensity})`);
  vignette.addColorStop(1, `rgba(0, 0, 0, ${0.76 * intensity})`);

  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, width, height);

  // Subtle grey particulate texture inside the fog.
  const grainAmount = Math.floor(380 * intensity);

  for (let i = 0; i < grainAmount; i++) {
    const x = Math.random() * width;
    const y = Math.random() * height;
    const alpha = randomRange(0.01, 0.05) * intensity;

    ctx.fillStyle = `rgba(214, 216, 218, ${alpha})`;
    ctx.fillRect(x, y, Math.random() > 0.8 ? 1.5 : 1, 1);
  }

  ctx.restore();
}

function drawTransition(time) {
  if (time < timeline.orbitStart) return;

  const morph = easeInOut(
    clamp((time - timeline.orbitStart) / 4.5)
  );

  drawOrbitalSystem(time, morph);

  if (time < timeline.orbitStart + 5) {
    ctx.save();
    ctx.globalAlpha = 1 - morph;
    drawSigilSeed(time);
    ctx.restore();
  }
}

function render(now) {
  const time = (now - startedAt) / 1000;

  setPhaseUI(time);
  drawBackground(time);

  if (time < timeline.orbitStart + 5) {
    const seedOpacity = 1 - easeInOut(clamp((time - 15) / 5));

    ctx.save();
    ctx.globalAlpha = seedOpacity;
    drawSigilSeed(time);
    ctx.restore();
  }

  drawTransition(time);

  if (time >= timeline.interactionStart) {
    drawVeins(time);
  }

  drawAtmosphere(time);

  requestAnimationFrame(render);
}

function pointerToCanvas(event) {
  const bounds = canvas.getBoundingClientRect();

  return {
    x: event.clientX - bounds.left,
    y: event.clientY - bounds.top
  };
}

canvas.addEventListener("pointerdown", (event) => {
  const time = (performance.now() - startedAt) / 1000;

  if (time < timeline.interactionStart) return;

  const point = pointerToCanvas(event);
  createVeinCluster(point.x, point.y);
});

window.addEventListener("resize", resize);

resize();
requestAnimationFrame(render);